/**
 * Pre-selects M neighbors per pixel for spatial resampling.
 *
 * Directly implements Algorithm 1 (paper §5).
 *
 * Samples K candidates from the precomputed R2 quasi-random disk.
 * Scores each using the CGNS compatibility heuristic (CGNSUtils.slang).
 * Selects M via weighted reservoir sampling (WRSReservoir.slang).
 *
 * Exactly one selection method is active at a time (enforced by the pass UI/properties).
 * The choice is passed as a compile-time define (StaticParams.slang -> kEnable*), so the
 * score dispatch below folds down to the single active scorer at compile time.
 *
 * Output: neighborSelections[linearizePixel(px) * M + i] = selected int2 neighbors for pixel i.
 */

import Scene.Scene;
import Utils.Sampling.TinyUniformSampleGenerator;
import CGNSUtils;
import WRSReservoir;
import Reservoir;
import Params;

#ifndef EPSILON
#define EPSILON 1e-8f
#endif

struct SelectNeighbors
{
    PathTracerParams params;

    Texture2D<float4>        cgnsGeometryGBuffer;   // (world normal xyz, cam-dist); dist==0 -> env hit
    Texture2D<float4>        cgnsIntegrandGBuffer;  // (integrand rgb, luminance); all zero -> env hit
    Texture1D<float2>        neighborOffsets;       // R2 quasi-random disk offsets
    RWStructuredBuffer<int2> neighborSelections;    // Output: M selected pixels per pixel (stride = gM)
    StructuredBuffer<PathReservoir> currReservoirs; // Used for p-value selection (running mean/variance/M of integrand luminance)

    float gGatherRadius;             // Radius for sampling neighbors
    int   gM;                        // M - selections to return (1 -> A-Chao, >1 -> A-ES)
    int   gNeighborCandidateCount;   // K — candidates scored per pixel
    int   gSpatialRoundId;           // Current reuse round if multiple are enabled

    float gScaleSolidAngle;    // Ω (sr); controls position falloff radius via s = sqrt(Ω/π) * queryDist
    float gNormalBeta;         // β: exponent on max(n·n', 0)

    bool gUseEarlyStopping;     // Stop candidate sampling once M candidates with score >= gEarlyStopCutoff are found (paper §5, §7.5)
    float gEarlyStopCutoff;     // Determines threshold for early stopping

    float gColorExponent;      // exponent applied to the color and luminance compatibility terms

    float3 computeWorldPos(int2 px, float dist)
    {
        float3 dir = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(uint2(px), params.frameDim));
        return gScene.camera.getPosition() + dir * dist;
    }

    /// Shared environment-hit handling: both invalid -> encourage reuse, mixed -> discourage.
    /// Returns true if the caller may proceed with the actual compatibility test.
    bool classifyHit(bool queryValid, bool candidateValid, out float score)
    {
        if (!queryValid && !candidateValid)
        {
            score = 1.0f; // both env map hits — encourage reuse
            return false;
        }
        if (!queryValid || !candidateValid)
        {
            score = EPSILON; // one surface, one env map — discourage
            return false;
        }
        score = 1.0f;
        return true;
    }

    float scoreCandidateGeometry(
        bool   queryValid,
        float3 queryPos,
        float3 queryNormal,
        float  queryScale,
        int2   candidatePx
    ) {
        float4 geometry       = cgnsGeometryGBuffer[candidatePx];
        float  candidateDist  = geometry.w;
        bool   candidateValid = candidateDist > 0.0f;
        float3 candidatePos   = candidateValid ? computeWorldPos(candidatePx, candidateDist) : float3(0, 0, 0);

        float score;
        if (!classifyHit(queryValid, candidateValid, score)) return score;

        return computeNeighborWeightGeometry(
            queryValid, candidateValid,
            queryPos, queryNormal,
            candidatePos, geometry.xyz,
            queryScale, gNormalBeta
        );
    }

    float scoreCandidateIntegrand(
        bool   queryValid,
        float3 queryColor,
        float  queryLuminance,
        int2   candidatePx
    ) {
        bool candidateValid = cgnsGeometryGBuffer[candidatePx].w > 0.0f;

        float score;
        if (!classifyHit(queryValid, candidateValid, score)) return score;

        float4 candidateIntegrand = cgnsIntegrandGBuffer[candidatePx];

        return computeNeighborWeightIntegrand(
            queryValid, candidateValid,
            queryColor, queryLuminance,
            candidateIntegrand.rgb, candidateIntegrand.a,
            gColorExponent
        );
    }

    float scoreCandidateIntegrandPValue(
        bool                 queryValid,
        const PathReservoir  queryReservoir,
        int2                 candidatePx
    ) {
        bool candidateValid = cgnsGeometryGBuffer[candidatePx].w > 0.0f;

        float score;
        if (!classifyHit(queryValid, candidateValid, score)) return score;

        PathReservoir candidateReservoir = currReservoirs[params.linearizePixel(candidatePx)];

        return computeIntegrandPValueScore(queryReservoir, candidateReservoir);
    }

    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;
        uint centralOffset = params.linearizePixel(pixel);
        PathReservoir queryReservoir = currReservoirs[centralOffset];

        // Basic random seed, unique from other passes
        uint seed = (params.useFixedSeed > 0) ? params.fixedSeed + 4 : kNumRenderPasses * (params.frameCount + params.seed) + 1337 * uint(gSpatialRoundId) + 4;
        var sg = TinyUniformSampleGenerator(pixel, seed);

        float4 qGeometryBuf = cgnsGeometryGBuffer[pixel];
        float4 qIntegrandBuf = cgnsIntegrandGBuffer[pixel];
        float3 queryNormal    = qGeometryBuf.xyz;
        float  queryDist      = qGeometryBuf.w;
        bool   queryValid     = queryDist > 0.0f;
        float3 queryPos       = queryValid ? computeWorldPos(int2(pixel), queryDist) : float3(0, 0, 0);
        float  queryScale     = computeNeighborScale(gScaleSolidAngle, queryDist);
        float3 queryColor     = qIntegrandBuf.rgb;
        float  queryLuminance = qIntegrandBuf.a;

        uint startIndex = uint(sampleNext1D(sg) * kSpatialNeighborSamples);
        uint base       = centralOffset * uint(gM);

        if (gM == 1)
        {
            // A-Chao WRS algorithm is simplest for M=1
            AChaoReservoir res;
            res.init(int2(pixel));

            for (int k = 0; k < gNeighborCandidateCount; k++)
            {
                int2 candidate = int2(round(float2(pixel)
                    + gGatherRadius * neighborOffsets[(startIndex + uint(k)) % kSpatialNeighborSamples]));
                if (!params.validateIntegerPixelBounds(candidate)) continue;

                float score = 1.0f;
                if (kEnableGeometrySelection) {
                    score = scoreCandidateGeometry(queryValid, queryPos, queryNormal, queryScale, candidate);
                } else if (kEnableIntegrandSelection) {
                    score = scoreCandidateIntegrand(queryValid, queryColor, queryLuminance, candidate);
                } else if (kEnableIntegrandPValueSelection) {
                    score = scoreCandidateIntegrandPValue(queryValid, queryReservoir, candidate);
                }

                float u     = sampleNext1D(sg);
                res.tryUpdate(candidate, score, u);

                if (gUseEarlyStopping && score >= gEarlyStopCutoff) break;
            }

            neighborSelections[base] = res.get();
        }
        else
        {
            // A-ES: key = log(u)/w per candidate, maintain top-M. Unbiased for any M.
            AESReservoir res;
            res.init(int2(pixel), uint(gM));

            int highQualityCount = 0;
            for (int k = 0; k < gNeighborCandidateCount; k++)
            {
                int2 candidate = int2(round(float2(pixel)
                    + gGatherRadius * neighborOffsets[(startIndex + uint(k)) % kSpatialNeighborSamples]));
                if (!params.validateIntegerPixelBounds(candidate)) continue;

                float score = 1.0f;
                if (kEnableGeometrySelection) {
                    score = scoreCandidateGeometry(queryValid, queryPos, queryNormal, queryScale, candidate);
                } else if (kEnableIntegrandSelection) {
                    score = scoreCandidateIntegrand(queryValid, queryColor, queryLuminance, candidate);
                } else if (kEnableIntegrandPValueSelection) {
                    score = scoreCandidateIntegrandPValue(queryValid, queryReservoir, candidate);
                }

                float u     = sampleNext1D(sg);
                res.tryUpdate(candidate, score, u, uint(gM));

                if (gUseEarlyStopping && score >= gEarlyStopCutoff)
                {
                    if (++highQualityCount >= gM) break;
                }
            }

            for (uint j = 0; j < uint(gM); j++)
            {
                neighborSelections[base + j] = res.get(j);
            }
        }
    }
}

cbuffer CB
{
    SelectNeighbors gSelectNeighbors;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSelectNeighbors.execute(dispatchThreadId.xy);
}
