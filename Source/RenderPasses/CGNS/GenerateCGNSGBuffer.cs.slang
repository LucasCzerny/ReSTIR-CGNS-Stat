/**
 * Generates the per-pixel G-buffers for neighbor selection.
 * Reads the V-buffer and writes textures consumed by SelectNeighbors.cs.slang:
 *   - cgnsGeometryGBuffer:  (world normal xyz, camera-to-surface distance)   RGBA16F
 *   - cgnsIntegrandGBuffer:     (integrand rgb, luminance)                       RGBA16F
 *   - cgnsMaterialGBuffer:  (materialID, roughness, specular F0 lum, emissive lum)  RGBA32F
 *   - cgnsBaseColorGBuffer: (diffuse albedo rgb, unused)                     RGBA16F
 *   - cgnsIdGBuffer:        (instanceID, meshID, primitiveID, unused)        RGBA32Uint
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
    RWTexture2D<float4> cgnsMaterialGBuffer;   // RGBA32F: (materialID, roughness, specular, emissive)
    RWTexture2D<float4> cgnsBaseColorGBuffer;  // RGBA16F: (diffuse albedo rgb, unused)
    RWTexture2D<uint4>  cgnsIdGBuffer;         // RGBA32Uint: (instanceID, meshID, primitiveID, unused)
    RWStructuredBuffer<PathReservoir> currReservoirs;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        HitInfo hit = HitInfo(vbuffer[pixel]);

        if (!hit.isValid())
        {
            cgnsGeometryGBuffer[pixel] = float4(0, 0, 0, 0);  // dist == 0 -> env hit
            cgnsIntegrandGBuffer[pixel] = float4(0, 0, 0, 0);
            cgnsMaterialGBuffer[pixel] = float4(0, 0, 0, 0);
            cgnsBaseColorGBuffer[pixel] = float4(0, 0, 0, 0);
            cgnsIdGBuffer[pixel] = uint4(0, 0, 0, 0);
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

        // Material properties. Specular and emissive are reduced to luminance scalars to fit one float4.
        ExplicitLodTextureSampler lod = ExplicitLodTextureSampler(0.f);
        uint hints = getMaterialInstanceHints(hit, true);
        IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties props = mi.getProperties(sd);

        cgnsMaterialGBuffer[pixel] = float4(
            float(sd.materialID),
            props.roughness,
            luminance(max(props.specularReflectance, float3(0.0f))),
            luminance(max(props.emission, float3(0.0f)))
        );
        cgnsBaseColorGBuffer[pixel] = float4(props.diffuseReflectionAlbedo, 0.0f);

        // Object identity. Stored as raw IDs (uint4) so comparisons are exact.
        GeometryInstanceID instanceID = hit.getInstanceID();
        cgnsIdGBuffer[pixel] = uint4(
            instanceID.index,
            gScene.getGeometryInstance(instanceID).geometryID,
            hit.getPrimitiveIndex(),
            0u
        );
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
