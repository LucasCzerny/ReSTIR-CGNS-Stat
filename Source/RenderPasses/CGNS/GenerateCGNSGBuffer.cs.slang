/**
 * Generates the per-pixel G-buffers for neighbor selection.
 * Reads the V-buffer and writes textures consumed by SelectNeighbors.cs.slang:
 *   - cgnsGeometryGBuffer:  (world normal xyz, camera-to-surface distance)   RGBA16F
 *   - cgnsIntegrandGBuffer: (integrand rgb, luminance)                       RGBA16F
 *
 * A zero geometry distance encodes an environment/invalid hit - SelectNeighbors uses
 * this sentinel to determine candidateValid.
 */

import Scene.Scene;
import Scene.HitInfo;
import Rendering.Materials.IMaterialInstance;
import LoadShadingData;
import Params;
import Reservoir;
import Utils.Color.ColorHelpers;

struct GenerateCGNSGBuffer
{
    PathTracerParams params;

    Texture2D<PackedHitInfo> vbuffer;
    RWTexture2D<float4> cgnsGeometryGBuffer;
    RWTexture2D<float4> cgnsIntegrandGBuffer;
    RWStructuredBuffer<PathReservoir> currReservoirs;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        HitInfo hit = HitInfo(vbuffer[pixel]);

        if (!hit.isValid())
        {
            cgnsGeometryGBuffer[pixel] = float4(0, 0, 0, 0);  // dist == 0 -> env hit
            cgnsIntegrandGBuffer[pixel] = float4(0, 0, 0, 0);
            return;
        }

        float3 cameraPos = gScene.camera.getPosition();
        float3 rayDir = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(pixel, params.frameDim));
        ShadingData sd = loadShadingData(hit, cameraPos, rayDir);
        float posDist = distance(cameraPos, sd.posW);

        cgnsGeometryGBuffer[pixel] = float4(sd.frame.N, posDist);

        const uint32_t reservoirIdx = params.linearizePixel(pixel);
        PathReservoir reservoir = currReservoirs[reservoirIdx];

        float3 integrand = float3(0.0);
        if (reservoir.totalWeight > 0.0f) {
            integrand = reservoir.integrand;
        }

        float lum = luminance(max(integrand, 0.0f));
        cgnsIntegrandGBuffer[pixel] = float4(integrand, lum);
    }
}

cbuffer CB
{
    GenerateCGNSGBuffer gGenerateCGNSGBuffer;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gGenerateCGNSGBuffer.execute(dispatchThreadId.xy);
}
