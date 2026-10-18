/**
 * Compatibility scoring for spatial neighbor selection.
 * Used by SelectNeighbors.cs.slang to score candidate pixels before WRS selection.
 *
 * Heuristic: h = h_position * h_normal  (paper §4.3)
 *   h_position = exp(-||x - y|| / s),      s = sqrt(Ω/π) * queryDist
 *   h_normal   = max(n_x · n_y, 0)^β
 *
 * Paper defaults: Ω = 0.05 sr, β = 8.
 */

 import Reservoir;

#ifndef EPSILON
#define EPSILON 1e-8f
#endif

static const float kPi = 3.14159265f;

// s = sqrt(Ω/π) * d  (characteristic world-space radius at distance d, paper §4.3)
// Controls position falloff radius ensuring the heuristic is agnostic to scene scale.
float computeNeighborScale(float solidAngle, float dist)
{
    return max(EPSILON, sqrt(solidAngle / kPi) * dist);
}

float computeNeighborWeightGeometry(
    bool   queryValid,
    bool   candidateValid,
    float3 queryPos,
    float3 queryNormal,
    float3 candidatePos,
    float3 candidateNormal,
    float  scale,      // s = sqrt(Ω/π) * queryDist
    float  normalBeta  // β: exponent on the normal compatibility term
)
{
    float positionScore = exp(-distance(queryPos, candidatePos) / scale);
    float normalScore   = max(0.0f, dot(queryNormal, candidateNormal));

    return clamp(positionScore * pow(normalScore, normalBeta), EPSILON, 1.0f);
}

float computeNeighborWeightIntegrand(
    bool   queryValid,
    bool   candidateValid,
    float3 queryColor,
    float  queryLuminance,
    float3 candidateColor,
    float  candidateLuminance,
    float  exponent
)
{
    // TODO: should the color comparison be changed to match the non-linear human perception more closely?
    float colorRelativeDiff = length(queryColor - candidateColor) / (length(queryColor) + length(candidateColor) + EPSILON);
    float colorScore = pow(1.0f - colorRelativeDiff, exponent);

    float luminanceRelativeDiff = abs(queryLuminance - candidateLuminance) / (queryLuminance + candidateLuminance + EPSILON);
    float luminanceScore = pow(1.0f - luminanceRelativeDiff, exponent);

    return clamp(colorScore * luminanceScore, EPSILON, 1.0f);
}

float computeIntegrandPValueScore(const PathReservoir queryReservoir, const PathReservoir candidateReservoir)
{
    if (queryReservoir.M < 1.0f || candidateReservoir.M < 1.0f)
        return EPSILON;

    return clamp(queryReservoir.computePValue(candidateReservoir), EPSILON, 1.0f);
}


