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

static const float3 kLuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);

float computeNeighborWeightIntegrand(
    bool   queryValid,
    bool   candidateValid,
    float3 queryPos,
    float3 queryIntegrand,
    float3 candidatePos,
    float3 candidateIntegrand,
    float  scale,       // s = sqrt(Ω/π) * queryDist
    float  exponent
)
{
    float queryLuminance = dot(max(queryIntegrand, 0.0f), kLuminanceWeights);
    float candidateLuminance = dot(max(candidateIntegrand, 0.0f), kLuminanceWeights);

    float relativeDiff = abs(queryLuminance - candidateLuminance) / (queryLuminance + candidateLuminance + EPSILON);

    float integrandScore = pow(1.0f - relativeDiff, exponent);

    float positionScore = exp(-distance(queryPos, candidatePos) / scale);

    return clamp(positionScore * integrandScore, EPSILON, 1.0f);
}
