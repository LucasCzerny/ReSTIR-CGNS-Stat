/***************************************************************************
 # Copyright (c) 2015-25, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import ReconnectionData;

import Params;

struct CandidateReservoir
{
    float totalWeight;  // Total weight of the samples in the reservoir.
    float3 integrand;   // The selected integrand.

    // Running mean and variance over the integrands of the input samples
    float mean;
    float variance;
    float M;

    ReconnectionData selectedReconnection; // This must be managed manually for now.

    // Referenced from PathReservoir::addNEEPath()
    [mutating]
    bool addVertex(inout SampleGenerator sg, float sampleMIS, float3 sampleIntegrand)
    {
        float samplePHat = luminance(sampleIntegrand);
        float sampleWeight = sampleMIS * samplePHat; // Subpaths are sampled as disjoint domains so m_i = 1.
        sampleWeight = isnan(sampleWeight) ? 0.0f : sampleWeight;

        this.totalWeight += sampleWeight;
        // print("Added Weight", sampleWeight);

        float rng = sampleNext1D(sg);
        bool selected = (rng * this.totalWeight < sampleWeight);
        this.integrand = selected ? sampleIntegrand : this.integrand;

        if (kEnableIntegrandPValueSelection) {
            float invM = 1.0 / M;

            float prevMean = mean;
            mean = prevMean + (samplePHat - prevMean) * invM;

            float prevVariance = variance;
            variance = (
                prevVariance
                + ((samplePHat - prevMean) * (samplePHat - mean) - prevVariance)
                * invM
            );
        }

        return selected;
    }

    float computeUCW()
    {
        float pHat = luminance(this.integrand);
        return (pHat == 0.0) ? 0.0f : this.totalWeight / pHat;
    }

    __init()
    {
        this.totalWeight = 0.0f;
        this.integrand = float3(0.0f);
        this.mean = 0.0f;
        this.variance = 0.0f;
        this.M = 0.0f;
        this.selectedReconnection = ReconnectionData();
    }
}

struct PathReservoir
{
    static const float confidenceCap = 20;

    float confidence;         // Sample history in the reservoir
    float totalWeight;        // Total weight
    float3 integrand;         // Radiance of selected sample (f / p in PSS).
    float2 subPixel;          // Associated subpixel in the pixel.

    // Running mean and variance over the integrands of the input samples
    float mean;
    float variance;
    float M;

    [mutating]
    bool addSampleFromReservoir(inout SampleGenerator sg, float mis, float3 integrand, float jacobian, const PathReservoir other)
    {
        float pHat = luminance(integrand);
        float w = mis * pHat * other.computeUCW() * jacobian;
        this.totalWeight = this.totalWeight + w;
        this.confidence = min(this.confidence + other.confidence, PathReservoir::confidenceCap);

        float rng = sampleNext1D(sg);
        bool selected = (rng * this.totalWeight < w);
        this.integrand = selected ? integrand : this.integrand;
        this.subPixel = selected ? other.subPixel : this.subPixel;

        if (kEnableIntegrandPValueSelection) {
            float invM = 1.0 / M;

            float prevMean = mean;
            mean = prevMean + (pHat - prevMean) * invM;

            float prevVariance = variance;
            variance = (
                prevVariance
                + ((pHat - prevMean) * (pHat - mean) - prevVariance)
                * invM
            );
        }

        return selected;
    }

    [mutating]
    bool add(inout SampleGenerator sg, float mis, CandidateReservoir candidateReservoir)
    {
        float weight = mis * candidateReservoir.totalWeight; // The pHat cancels out with the 1 / pHat from the UCW.
        this.totalWeight = this.totalWeight + weight;
        this.confidence = min(this.confidence + 1, PathReservoir::confidenceCap);

        float rng = sampleNext1D(sg);
        bool selected = (rng * this.totalWeight < weight);
        this.integrand = selected ? candidateReservoir.integrand : this.integrand;

        if (kEnableIntegrandPValueSelection) {
            float totalM = max(M + candidateReservoir.M, 1e-5);
            float invTotalM = 1.0 / totalM;

            float candidateReservoirRatio = candidateReservoir.M * invTotalM;
            float totalMean = mean + (candidateReservoir.mean - mean) * candidateReservoirRatio;

            float meanDiff = mean - totalMean;
            float candidateMeanDiff = candidateReservoir.mean - totalMean;

            float SSE = M * variance + candidateReservoir.M * candidateReservoir.variance;
            float SSG = M * meanDiff * meanDiff + candidateReservoir.M * candidateMeanDiff * candidateMeanDiff;
            float SST = SSE + SSG;
            variance = SST * invTotalM;

            mean = totalMean;
            M = totalM;
        }

        return selected;
    }

    float computeUCW()
    {
        float pHat = luminance(this.integrand);
        return (pHat == 0.0) ? 0.0f : this.totalWeight / pHat;
    }

    [mutating]
    void setSubPixel(const float2 subPixel)
    {
        this.subPixel = subPixel;
    }

    float erf(float x)
    {
        float t = 1.0 / (1.0 + 0.3275911 * abs(x));
        float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
        return sign(x) * y;
    }

    float computePValue(PathReservoir other)
    {
        float se = sqrt(variance / M + other.variance / other.M);
        if (se <= 0.0)
            return (mean == other.mean) ? 1.0 : 0.0;

        // z score
        float z = (mean - other.mean) / se;
        return 1.0 - erf(abs(z) / sqrt(2.0));
    }

    __init()
    {
        this.confidence = 0;
        this.totalWeight = 0.0f;
        this.integrand = float3(0.0f);
        this.subPixel = float2(0.5f, 0.5f);
        this.mean = 0.0f;
        this.variance = 0.0f;
        this.M = 0.0f;
    }
};
