// Unity built-in shader source. Copyright (c) 2016 Unity Technologies. MIT license (see license.txt)

/*
MODIFICATIONS MADE UNDER...
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Upgrade NOTE: excluded shader from DX11 because it uses wrong array syntax (type[size] name)
#pragma exclude_renderers d3d11

#ifndef UNITY_STANDARD_BRDF_INCLUDED
#define UNITY_STANDARD_BRDF_INCLUDED

#include "UnityCG.cginc"

#include "UnityStandardConfig.cginc"
#include "UnityLightingCommon.cginc"




//-----------------------------------------------------------------------------
// Helper to convert smoothness to roughness
//-----------------------------------------------------------------------------

float PerceptualRoughnessToRoughness(float perceptualRoughness)
{
    return perceptualRoughness * perceptualRoughness;
}

half RoughnessToPerceptualRoughness(half roughness)
{
    return sqrt(roughness);
}

// Smoothness is the user facing name
// it should be perceptualSmoothness but we don't want the user to have to deal with this name
half SmoothnessToRoughness(half smoothness)
{
    return (1 - smoothness) * (1 - smoothness);
}

float SmoothnessToPerceptualRoughness(float smoothness)
{
    return (1 - smoothness);
}

//-------------------------------------------------------------------------------------

inline half Pow4 (half x)
{
    return x*x*x*x;
}

inline float2 Pow4 (float2 x)
{
    return x*x*x*x;
}

inline half3 Pow4 (half3 x)
{
    return x*x*x*x;
}

inline half4 Pow4 (half4 x)
{
    return x*x*x*x;
}

// Pow5 uses the same amount of instructions as generic pow(), but has 2 advantages:
// 1) better instruction pipelining
// 2) no need to worry about NaNs
inline half Pow5 (half x)
{
    return x*x * x*x * x;
}

inline half2 Pow5 (half2 x)
{
    return x*x * x*x * x;
}

inline half3 Pow5 (half3 x)
{
    return x*x * x*x * x;
}

inline half4 Pow5 (half4 x)
{
    return x*x * x*x * x;
}

inline half3 FresnelTerm (half3 F0, half cosA)
{
    half t = Pow5 (1 - cosA);   // ala Schlick interpoliation
    return F0 + (1-F0) * t;
}
inline half3 FresnelLerp (half3 F0, half3 F90, half cosA)
{
    half t = Pow5 (1 - cosA);   // ala Schlick interpoliation
    return lerp (F0, F90, t);
}
// approximage Schlick with ^4 instead of ^5
inline half3 FresnelLerpFast (half3 F0, half3 F90, half cosA)
{
    half t = Pow4 (1 - cosA);
    return lerp (F0, F90, t);
}

// Note: Disney diffuse must be multiply by diffuseAlbedo / PI. This is done outside of this function.
half DisneyDiffuse(half NdotV, half NdotL, half LdotH, half perceptualRoughness)
{
	//return 0.0h;
    half fd90 = 0.5 + 2 * LdotH * LdotH * perceptualRoughness;
    // Two schlick fresnel term
    half lightScatter   = (1 + (fd90 - 1) * Pow5(1 - NdotL));
    half viewScatter    = (1 + (fd90 - 1) * Pow5(1 - NdotV));

    return lightScatter * viewScatter;
}

// NOTE: Visibility term here is the full form from Torrance-Sparrow model, it includes Geometric term: V = G / (N.L * N.V)
// This way it is easier to swap Geometric terms and more room for optimizations (except maybe in case of CookTorrance geom term)

// Generic Smith-Schlick visibility term
inline half SmithVisibilityTerm (half NdotL, half NdotV, half k)
{
    half gL = NdotL * (1-k) + k;
    half gV = NdotV * (1-k) + k;
    return 1.0 / (gL * gV + 1e-5f); // This function is not intended to be running on Mobile,
                                    // therefore epsilon is smaller than can be represented by half
}

// Smith-Schlick derived for Beckmann
inline half SmithBeckmannVisibilityTerm (half NdotL, half NdotV, half roughness)
{
    half c = 0.797884560802865h; // c = sqrt(2 / Pi)
    half k = roughness * c;
    return SmithVisibilityTerm (NdotL, NdotV, k) * 0.25f; // * 0.25 is the 1/4 of the visibility term
}

// Ref: http://jcgt.org/published/0003/02/03/paper.pdf
inline float SmithJointGGXVisibilityTerm (float NdotL, float NdotV, float roughness)
{
#if 1
    // Original formulation:
    //  lambda_v    = (-1 + sqrt(a2 * (1 - NdotL2) / NdotL2 + 1)) * 0.5f;
    //  lambda_l    = (-1 + sqrt(a2 * (1 - NdotV2) / NdotV2 + 1)) * 0.5f;
    //  G           = 1 / (1 + lambda_v + lambda_l);

    // Reorder code to be more optimal
    half a          = roughness; 
    half a2         = a * a;

    half lambdaV    = NdotL * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
    half lambdaL    = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);

    // Simplify visibility term: (2.0f * NdotL * NdotV) /  ((4.0f * NdotL * NdotV) * (lambda_v + lambda_l + 1e-5f));
    return 0.5f / (lambdaV + lambdaL + 1e-5f);  // This function is not intended to be running on Mobile,
                                                // therefore epsilon is smaller than can be represented by half
#else
    // Approximation of the above formulation (simplify the sqrt, not mathematically correct but close enough)
    float a = roughness;
    float lambdaV = NdotL * (NdotV * (1 - a) + a);
    float lambdaL = NdotV * (NdotL * (1 - a) + a);

#if defined(SHADER_API_SWITCH)
    return 0.5f / (lambdaV + lambdaL + 1e-4f); // work-around against hlslcc rounding error
#else
    return 0.5f / (lambdaV + lambdaL + 1e-5f);
#endif

#endif
}

inline float GGXTerm (float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float d = (NdotH * a2 - NdotH) * NdotH + 1.0f; // 2 mad
    return UNITY_INV_PI * a2 / (d * d + 1e-7f); // This function is not intended to be running on Mobile,
                                            // therefore epsilon is smaller than what can be represented by half
}





inline half PerceptualRoughnessToSpecPower (half perceptualRoughness)
{
    half m = PerceptualRoughnessToRoughness(perceptualRoughness);   // m is the true academic roughness.
    half sq = max(1e-4f, m*m);
    half n = (2.0 / sq) - 2.0;                          // https://dl.dropboxusercontent.com/u/55891920/papers/mm_brdf.pdf
    n = max(n, 1e-4f);                                  // prevent possible cases of pow(0,0), which could happen when roughness is 1.0 and NdotH is zero
    return n;
}

// BlinnPhong normalized as normal distribution function (NDF)
// for use in micro-facet model: spec=D*G*F
// eq. 19 in https://dl.dropboxusercontent.com/u/55891920/papers/mm_brdf.pdf
inline half NDFBlinnPhongNormalizedTerm (half NdotH, half n)
{
    // norm = (n+2)/(2*pi)
    half normTerm = (n + 2.0) * (0.5/UNITY_PI);

    half specTerm = pow (NdotH, n);
    return specTerm * normTerm;
}

//-------------------------------------------------------------------------------------
/*
// https://s3.amazonaws.com/docs.knaldtech.com/knald/1.0.0/lys_power_drops.html

const float k0 = 0.00098, k1 = 0.9921;
// pass this as a constant for optimization
const float fUserMaxSPow = 100000; // sqrt(12M)
const float g_fMaxT = ( exp2(-10.0/fUserMaxSPow) - k0)/k1;
float GetSpecPowToMip(float fSpecPow, int nMips)
{
   // Default curve - Inverse of TB2 curve with adjusted constants
   float fSmulMaxT = ( exp2(-10.0/sqrt( fSpecPow )) - k0)/k1;
   return float(nMips-1)*(1.0 - clamp( fSmulMaxT/g_fMaxT, 0.0, 1.0 ));
}

    //float specPower = PerceptualRoughnessToSpecPower(perceptualRoughness);
    //float mip = GetSpecPowToMip (specPower, 7);
*/

inline float3 Unity_SafeNormalize(float3 inVec)
{
    float dp3 = max(0.001f, dot(inVec, inVec));
    return inVec * rsqrt(dp3);
}

//-------------------------------------------------------------------------------------

// Note: BRDF entry points use smoothness and oneMinusReflectivity for optimization
// purposes, mostly for DX9 SM2.0 level. Most of the math is being done on these (1-x) values, and that saves
// a few precious ALU slots.


// Main Physically Based BRDF
// Derived from Disney work and based on Torrance-Sparrow micro-facet model
//
//   BRDF = kD / pi + kS * (D * V * F) / 4
//   I = BRDF * NdotL
//
// * NDF (depending on UNITY_BRDF_GGX):
//  a) Normalized BlinnPhong
//  b) GGX
// * Smith for Visiblity term
// * Schlick approximation for Fresnel
half4 BRDF1_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi)
{
    float perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);

// NdotV should not be negative for visible pixels, but it can happen due to perspective projection and normal mapping
// In this case normal should be modified to become valid (i.e facing camera) and not cause weird artifacts.
// but this operation adds few ALU and users may not want it. Alternative is to simply take the abs of NdotV (less correct but works too).
// Following define allow to control this. Set it to 0 if ALU is critical on your platform.
// This correction is interesting for GGX with SmithJoint visibility function because artifacts are more visible in this case due to highlight edge of rough surface
// Edit: Disable this code by default for now as it is not compatible with two sided lighting used in SpeedTree.
#define UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV 0

#if UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV
    // The amount we shift the normal toward the view vector is defined by the dot product.
    half shiftAmount = dot(normal, viewDir);
    normal = shiftAmount < 0.0f ? normal + viewDir * (-shiftAmount + 1e-5f) : normal;
    // A re-normalization should be applied here but as the shift is small we don't do it to save ALU.
    //normal = normalize(normal);

    float nv = saturate(dot(normal, viewDir)); // TODO: this saturate should no be necessary here
#else
    half nv = abs(dot(normal, viewDir));    // This abs allow to limit artifact
#endif

    float nl = saturate(dot(normal, light.dir));
    float nh = saturate(dot(normal, halfDir));

    half lv = saturate(dot(light.dir, viewDir));
    half lh = saturate(dot(light.dir, halfDir));

    // Diffuse term
    half diffuseTerm = DisneyDiffuse(nv, nl, lh, perceptualRoughness) * nl;

    // Specular term
    // HACK: theoretically we should divide diffuseTerm by Pi and not multiply specularTerm!
    // BUT 1) that will make shader look significantly darker than Legacy ones
    // and 2) on engine side "Non-important" lights have to be divided by Pi too in cases when they are injected into ambient SH
    float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
#if UNITY_BRDF_GGX
    // GGX with roughtness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughtness remapping.
    roughness = max(roughness, 0.002);
    float V = SmithJointGGXVisibilityTerm (nl, nv, roughness);
    float D = GGXTerm (nh, roughness);
#else
    // Legacy
    half V = SmithBeckmannVisibilityTerm (nl, nv, roughness);
    half D = NDFBlinnPhongNormalizedTerm (nh, PerceptualRoughnessToSpecPower(perceptualRoughness));
#endif

    float specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later

#   ifdef UNITY_COLORSPACE_GAMMA
        specularTerm = sqrt(max(1e-4h, specularTerm));
#   endif

    // specularTerm * nl can be NaN on Metal in some cases, use max() to make sure it's a sane value
    specularTerm = max(0, specularTerm * nl);
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = 0.0;
#endif

    // surfaceReduction = Int D(NdotH) * NdotH * Id(NdotL>0) dH = 1/(roughness^2+1)
    half surfaceReduction;
#   ifdef UNITY_COLORSPACE_GAMMA
        surfaceReduction = 1.0-0.28*roughness*perceptualRoughness;      // 1-0.28*x^3 as approximation for (1/(x^4+1))^(1/2.2) on the domain [0;1]
#   else
        surfaceReduction = 1.0 / (roughness*roughness + 1.0);           // fade \in [0.5;1]
#   endif

    // To provide true Lambert lighting, we need to be able to kill specular completely.
    specularTerm *= any(specColor) ? 1.0 : 0.0;

    half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));
    half3 color =   diffColor * (gi.diffuse + light.color * diffuseTerm)
                    + specularTerm * light.color * FresnelTerm (specColor, lh)
                    + surfaceReduction * gi.specular * FresnelLerp (specColor, grazingTerm, nv);

    return half4(color, 1);
}

// Based on Minimalist CookTorrance BRDF
// Implementation is slightly different from original derivation: http://www.thetenthplanet.de/archives/255
//
// * NDF (depending on UNITY_BRDF_GGX):
//  a) BlinnPhong
//  b) [Modified] GGX
// * Modified Kelemen and Szirmay-​Kalos for Visibility term
// * Fresnel approximated with 1/LdotH
half4 BRDF2_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi)
{
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);

    half nl = saturate(dot(normal, light.dir));
    float nh = saturate(dot(normal, halfDir));
    half nv = saturate(dot(normal, viewDir));
    float lh = saturate(dot(light.dir, halfDir));

    // Specular term
    half perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

#if UNITY_BRDF_GGX

    // GGX Distribution multiplied by combined approximation of Visibility and Fresnel
    // See "Optimizing PBR for Mobile" from Siggraph 2015 moving mobile graphics course
    // https://community.arm.com/events/1155
    half a = roughness;
    float a2 = a*a;

    float d = nh * nh * (a2 - 1.f) + 1.00001f;
#ifdef UNITY_COLORSPACE_GAMMA
    // Tighter approximation for Gamma only rendering mode!
    // DVF = sqrt(DVF);
    // DVF = (a * sqrt(.25)) / (max(sqrt(0.1), lh)*sqrt(roughness + .5) * d);
    float specularTerm = a / (max(0.32f, lh) * (1.5f + roughness) * d);
#else
    float specularTerm = a2 / (max(0.1f, lh*lh) * (roughness + 0.5f) * (d * d) * 4);
#endif

    // on mobiles (where half actually means something) denominator have risk of overflow
    // clamp below was added specifically to "fix" that, but dx compiler (we convert bytecode to metal/gles)
    // sees that specularTerm have only non-negative terms, so it skips max(0,..) in clamp (leaving only min(100,...))
#if defined (SHADER_API_MOBILE)
    specularTerm = specularTerm - 1e-4f;
#endif

#else

    // Legacy
    half specularPower = PerceptualRoughnessToSpecPower(perceptualRoughness);
    // Modified with approximate Visibility function that takes roughness into account
    // Original ((n+1)*N.H^n) / (8*Pi * L.H^3) didn't take into account roughness
    // and produced extremely bright specular at grazing angles

    half invV = lh * lh * smoothness + perceptualRoughness * perceptualRoughness; // approx ModifiedKelemenVisibilityTerm(lh, perceptualRoughness);
    half invF = lh;

    half specularTerm = ((specularPower + 1) * pow (nh, specularPower)) / (8 * invV * invF + 1e-4h);

#ifdef UNITY_COLORSPACE_GAMMA
    specularTerm = sqrt(max(1e-4f, specularTerm));
#endif

#endif

#if defined (SHADER_API_MOBILE)
    specularTerm = clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
#endif
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = 0.0;
#endif

    // surfaceReduction = Int D(NdotH) * NdotH * Id(NdotL>0) dH = 1/(realRoughness^2+1)

    // 1-0.28*x^3 as approximation for (1/(x^4+1))^(1/2.2) on the domain [0;1]
    // 1-x^3*(0.6-0.08*x)   approximation for 1/(x^4+1)
#ifdef UNITY_COLORSPACE_GAMMA
    half surfaceReduction = 0.28;
#else
    half surfaceReduction = (0.6-0.08*perceptualRoughness);
#endif

    surfaceReduction = 1.0 - roughness*perceptualRoughness*surfaceReduction;

    half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));
    half3 color =   (diffColor + specularTerm * specColor) * light.color * nl
                    + gi.diffuse * diffColor
                    + surfaceReduction * gi.specular * FresnelLerpFast (specColor, grazingTerm, nv);

    return half4(color, 1);
}

sampler2D_float unity_NHxRoughness;
half3 BRDF3_Direct(half3 diffColor, half3 specColor, half rlPow4, half smoothness)
{
    half LUT_RANGE = 16.0; // must match range in NHxRoughness() function in GeneratedTextures.cpp
    // Lookup texture to save instructions
    half specular = tex2D(unity_NHxRoughness, half2(rlPow4, SmoothnessToPerceptualRoughness(smoothness))).r * LUT_RANGE;
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specular = 0.0;
#endif

    return diffColor + specular * specColor;
}

half3 BRDF3_Indirect(half3 diffColor, half3 specColor, UnityIndirect indirect, half grazingTerm, half fresnelTerm)
{
    half3 c = indirect.diffuse * diffColor;
    c += indirect.specular * lerp (specColor, grazingTerm, fresnelTerm);
    return c;
}

// Old school, not microfacet based Modified Normalized Blinn-Phong BRDF
// Implementation uses Lookup texture for performance
//
// * Normalized BlinnPhong in RDF form
// * Implicit Visibility term
// * No Fresnel term
//
// TODO: specular is too weak in Linear rendering mode
half4 BRDF3_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi)
{
    float3 reflDir = reflect (viewDir, normal);

    half nl = saturate(dot(normal, light.dir));
    half nv = saturate(dot(normal, viewDir));

    // Vectorize Pow4 to save instructions
    half2 rlPow4AndFresnelTerm = Pow4 (float2(dot(reflDir, light.dir), 1-nv));  // use R.L instead of N.H to save couple of instructions
    half rlPow4 = rlPow4AndFresnelTerm.x; // power exponent must match kHorizontalWarpExp in NHxRoughness() function in GeneratedTextures.cpp
    half fresnelTerm = rlPow4AndFresnelTerm.y;

    half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));

    half3 color = BRDF3_Direct(diffColor, specColor, rlPow4, smoothness);
    color *= light.color * nl;
    color += BRDF3_Indirect(diffColor, specColor, gi, grazingTerm, fresnelTerm);

    return half4(color, 1);
}

//CT Definitions taken from https://graphicscompendium.com/references/cook-torrance
inline float CTTerm (float NdotH, float roughness)
{
    float ooa2 = 1.0f/(roughness*roughness);//one over alpha squared
	float NdotH2 = NdotH*NdotH;
	float negtan2 = (NdotH2-1)/NdotH2;
	float expo = negtan2*ooa2;
	float natbase = 2.71828182845904523536f;
	float d = UNITY_INV_PI * ooa2 * pow(natbase,expo) /  max(NdotH2 * NdotH2,0.001);
	//float d = 0.25f * ooa2 * pow(natbase,expo) / max(NdotH2 * NdotH2,0.001); //changed to use 1/4 instead of inv_pi
	return d;
}


//CT ported from Mitsuba
inline float CTTermMitsuba(float3 m, float roughness){
	if(dot(m, float3(0.0f,0.0f,1.0f)) <= 0.0f){return 0.0f;}
	float cosTheta = dot(m,float3(0.0,0.0,1.0));//added by me to save a dot product computation
	//return cosTheta;
	float cosTheta2 = cosTheta*cosTheta;
	float m_alphaU = roughness;
	float m_alphaV = roughness;
	float beckmannExponent = ((m[0]*m[0]) / (m_alphaU * m_alphaU) + (m[1]*m[1]) / (m_alphaV * m_alphaV)) / cosTheta2;
	float result = pow(2.71828182845904523536f,-beckmannExponent)/(3.14159f * m_alphaU * m_alphaV * cosTheta2 * cosTheta2);
	if (result * cosTheta < 1e-20f){result = 0.0f;}
	return result;
}

inline float AniCTTermMitsuba(float3 m, float ax, float ay){
	if(dot(m, float3(0.0f,0.0f,1.0f)) <= 0.0f){return 0.0f;}
	float cosTheta = dot(m,float3(0.0,0.0,1.0));//added by me to save a dot product computation
	//return cosTheta;
	float cosTheta2 = cosTheta*cosTheta;
	float m_alphaU = ax;
	float m_alphaV = ay;
	float beckmannExponent = ((m[0]*m[0]) / (m_alphaU * m_alphaU) + (m[1]*m[1]) / (m_alphaV * m_alphaV)) / cosTheta2;
	float result = pow(2.71828182845904523536f,-beckmannExponent)/(3.14159f * m_alphaU * m_alphaV * cosTheta2 * cosTheta2);
	if (result * cosTheta < 1e-20f){result = 0.0f;}
	return result;
}
/*Mitsuba Shadowing and Masking Function*/
float tanTheta(float3 v){
	float temp = 1.0f - v[2]*v[2];
	if(temp<0.0f){return 0.0f;}
	return sqrt(temp)/v[2];	
}

float mitsuba_smithG1_Beckmann(float3 v, float3 m, float alpha){
	float vm = dot(v, m);
	if(vm*v[2]<=0.0f){return 0.0f;}
	float tan_theta = abs(tanTheta(v));
	float a = 1.0f/(alpha*tan_theta);
	if(a>=1.6f){return 1.0f;}
	float a2 = a*a;
	return (3.535f * a + 2.181f * a2)/(1.0f + 2.276f * a + 2.577f * a2);
	
}

float mitsuba_smithG1(float3 v, float3 m, float alpha){
	float vm = dot(v, m);
	if(vm*v[2]<=0.0f){return 0.0f;}
	float tan_theta = abs(tanTheta(v));
	if(tan_theta==0.0f){return 1.0f;}
	float root = alpha * tan_theta;
	return 2.0f/(1.0f + sqrt(1.0f + root*root));
}

float mitsuba_G_Beckmann(float3 wi, float3 wo, float3 m, float alpha){
	return mitsuba_smithG1_Beckmann(wi,m,alpha) * mitsuba_smithG1_Beckmann(wo,m,alpha);
}

float mitsuba_G(float3 wi, float3 wo, float3 m, float alpha){
	return mitsuba_smithG1(wi,m,alpha) * mitsuba_smithG1(wo,m,alpha);
}

float mitsuba_Fresnel(float cosThetaI, float eta){
	//return eta;
	if(cosThetaI<0.0f){cosThetaI = abs(cosThetaI);}
	//if(eta < 5.0f){return 0.0f;}
	float k=0.0f;
	float cosThetaI2 = cosThetaI*cosThetaI;
    float sinThetaI2 = 1.0f-cosThetaI2;
    float sinThetaI4 = sinThetaI2*sinThetaI2;

    float temp1 = eta*eta - k*k - sinThetaI2;
    float a2pb2 = sqrt(temp1*temp1 + 4*k*k*eta*eta);
    float a     = sqrt(0.5f * (a2pb2 + temp1));

    float term1 = a2pb2 + cosThetaI2;
    float term2 = 2.0f*a*cosThetaI;

    float Rs2 = (term1 - term2) / (term1 + term2);

    float term3 = a2pb2*cosThetaI2 + sinThetaI4;
    float term4 = term2*sinThetaI2;

    float Rp2 = Rs2 * (term3 - term4) / (term3 + term4);

    return 0.5f * (Rp2 + Rs2);
	
}



/*For all versions*/

float _F0;

float _dr,_dg,_db,_sr,_sg,_sb,_RoughnessAlpha,_FresnelEta,_Anisotropic;

float _mitsuba_dist_mode;

/*For Simple Multiple Scattering*/

float _use_SMS;

sampler2D _Etable;
float _plot_mode;

// Used for MERL and INFORMATIVE_SHADER
/*MERL BRDF from Texture
  Based off of code taken
  from mitsuba.
*/

float3 cross_product (float3 v1, float3 v2)
{
	return float3(v1[1]*v2[2] - v1[2]*v2[1],v1[2]*v2[0] - v1[0]*v2[2],v1[0]*v2[1] - v1[1]*v2[0]);
}


// rotate vector along one axis
float3 rotate_vector(float3 ivector, float3 iaxis, float a)
{
	float3 outv;
	float temp;
	float3 cross;
	float cos_ang = cos(a);
	float sin_ang = sin(a);

	outv[0] = ivector[0] * cos_ang;
	outv[1] = ivector[1] * cos_ang;
	outv[2] = ivector[2] * cos_ang;

	temp = iaxis[0]*ivector[0]+iaxis[1]*ivector[1]+iaxis[2]*ivector[2];
	temp = temp*(1.0-cos_ang);

	outv[0] += iaxis[0] * temp;
	outv[1] += iaxis[1] * temp;
	outv[2] += iaxis[2] * temp;

	float3 icross = cross_product (iaxis,ivector);
	
	outv[0] += icross[0] * sin_ang;
	outv[1] += icross[1] * sin_ang;
	outv[2] += icross[2] * sin_ang;
	return outv;
}
//moved up for MSVG
//float Frame_cosTheta(float3 v){
//	return v[2];
//}

float Frame_sinTheta2(float3 v){
	return 1.0f - v[2] * v[2];
}

float Frame_sinTheta(float3 v){
	float tmp = Frame_sinTheta2(v);
	if(tmp<=0){return 0.0f;}
	return sqrt(tmp);
}

float Frame_sinPhi(float3 v){
	float sinTheta = Frame_sinTheta(v);
	if(sinTheta == 0.0f)
	{return 1.0f;}
	float unbounded_answer = v[1]/sinTheta;
	if(unbounded_answer>1.0f){return 1.0f;}
	if(unbounded_answer<-1.0f){return -1.0f;}
	return unbounded_answer;	
}

float Frame_cosPhi(float3 v){
	float sinTheta = Frame_sinTheta(v);
	if(sinTheta == 0.0f){return 1.0f;}
	float unbounded_answer = v[0]/sinTheta;
	if(unbounded_answer>1.0f){return 1.0f;}
	if(unbounded_answer<-1.0f){return -1.0f;}
	return unbounded_answer;	
}
/*This is ported from the MERL_BRDF.h that we all seem to have*/

float4 std_coords_to_half_diff_coords(float theta_in, float fi_in, float theta_out, float fi_out)
{

        // compute in vector
        float in_vec_z = cos(theta_in);
        float proj_in_vec = sin(theta_in);
        float in_vec_x = proj_in_vec*cos(fi_in);
        float in_vec_y = proj_in_vec*sin(fi_in);
        float3 vin= float3(in_vec_x,in_vec_y,in_vec_z);
        vin = Unity_SafeNormalize(vin);


        // compute out vector
        float out_vec_z = cos(theta_out);
        float proj_out_vec = sin(theta_out);
        float out_vec_x = proj_out_vec*cos(fi_out);
        float out_vec_y = proj_out_vec*sin(fi_out);
        float3 vout= float3(out_vec_x,out_vec_y,out_vec_z);
        vout = Unity_SafeNormalize(vout);


        // compute halfway vector
        float half_x = (in_vec_x + out_vec_x)/2.0f;
        float half_y = (in_vec_y + out_vec_y)/2.0f;
        float half_z = (in_vec_z + out_vec_z)/2.0f;
        float3 vhalf = float3(half_x,half_y,half_z);
        vhalf = Unity_SafeNormalize(vhalf);

        // compute  theta_half, fi_half
        float theta_half = acos(vhalf[2]);
        float fi_half = atan2(vhalf[1], vhalf[0]);


        float3 bi_normal = float3(0.0, 1.0, 0.0);
        float3 normal = float3(0.0, 0.0, 1.0);
        float3 temp;
        float3 diff;

        // compute diff vector
        temp = rotate_vector(vin, normal , -fi_half);
        diff = rotate_vector(temp, bi_normal, -theta_half);

        // compute  theta_diff, fi_diff
        float theta_diff = acos(diff[2]);
        float fi_diff = atan2(diff[1], diff[0]);
		return float4(theta_half,fi_half,theta_diff,fi_diff);
}



/* Constants from MERL code
float INV_PI = 0.31830988618f;
float BRDF_SAMPLING_RES_THETA_H = 89.0f;
float BRDF_SAMPLING_RES_THETA_D = 89.0f;
float BRDF_SAMPLING_RES_PHI_D = 360.0f;

float SAMPLING_RES_THETA_H = 89.0f;
float SAMPLING_RES_THETA_D = 89.0f;
float SAMPLING_RES_PHI_D = 360.0f;

float thetaHFactor = 2.0f * 0.31830988618f * 89.0f * 89.0f;
float thetaDFactor = 2.0f * 0.31830988618f * 89.0f;
float phiDFactor = 0.31830988618f * 360.0f / 2.0f;

For some reason, setting these as global variables does not work,
so I have hard coded the values below.
*/


float getThetaHIndex(float thetaH){
	if(thetaH <= 0.0f){return 89.0f;}
	if(thetaH >= 3.14159265359){return 89.0f;}
	return floor(sqrt(thetaH) * 0.7978845608f * 89.0f);// that first constant is sqrt(2/pi)
}

float getThetaDIndex(float thetaD){
	if (thetaD <= 0.0f){return 0.0f;}
    if (thetaD >= 3.14159265359 / 2.0f){return 89.0f;}
    return floor(thetaD * 2.0f * 0.31830988618f * 89.0f);
}

float getPhiDIndex(float phiD){
	if (phiD < 0.0f){phiD += 3.14159265359;}
    if (phiD <= 0.0){return 0.0f;}
    if (phiD >= 3.14159265359){return 179.0f;}
    return floor(phiD * 0.31830988618f * 179.0f);
}

float getThetaHIndex_raw(float thetaH){
	return (sqrt(thetaH) * 0.7978845608f * 89.0f);
}

float getThetaDIndex_raw(float thetaD){
    return (thetaD * 2.0f * 0.31830988618f * 89.0f);
}

float getPhiDIndex_raw(float phiD){
	if (phiD < 0.0f){phiD += 3.14159265359;}
    if (phiD <= 0.0){return 0.0f;}
    if (phiD >= 3.14159265359){return 179.0f;}
    return (phiD * 0.31830988618f * 179.0f);
}

float _show_info_shader;
//-----------------------------------------------------------------------------
// Informative Shader as a callable method
//-----------------------------------------------------------------------------
half3 INFORMATIVE_SHADER (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0)){
		
	if(_plot_mode>0.0f){
		//return half4(tex_xy[0],0,tex_xy[1],1);
		normal = float3(0,0,1);
		float halfTheta = (1-tex_xy[0])*UNITY_PI/2.0f;
		float halfDiff = (1-tex_xy[1])*UNITY_PI/2.0f;
		//float3 halfDir = float3(sin(halfTheta),0,cos(halfTheta));
		light.dir = normalize(float3(sin(halfDiff),0,cos(halfDiff)));
		viewDir = normalize(float3(-sin(halfDiff),0,cos(halfDiff)));
		normal = normalize(float3(0,sin(halfTheta),cos(halfTheta)));
		//light.color = float3(0.5,0.5,0.5);
	}
		
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 uvec;
	float3 vvec;
	float3 local_normal = float3(0,0,1);
	if(nz<0.0f)
	{
		float a = 1.0f / (1.0f - nz);
		float b = nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, -b, nx);
		vvec = float3(b, ny * ny*a - 1.0f, -ny);
	}
	else{
		float a = 1.0f / (1.0f + nz);
		float b = -nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, b, -nx);
		vvec = float3(b, 1.0f - ny * ny * a, -ny);
	}
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	float nl = dot(float3(0,0,1), local_light_dir);
	if(nl<0.0f) return float3(0,0,0);
	float nv = dot(float3(0,0,1), local_view_dir);
	
	float twi = acos(local_light_dir[2]);
    float two = acos(local_view_dir[2]);
    float pwi = atan2(local_light_dir[1], local_light_dir[0]);
    float pwo = atan2(local_view_dir[1],local_view_dir[0]);

	
	float4 thetaphiangles = std_coords_to_half_diff_coords(two,pwo,twi,pwi);
	float3 coordinates = float3(thetaphiangles[3],thetaphiangles[2],thetaphiangles[0]);
	//coordinates = getAnglesFromVect(local_light_dir,local_half_dir);
    //return float4(float3(getThetaHIndex(coordinates[2])/89.0f,getThetaDIndex(coordinates[1])/89.0f,getPhiDIndex(coordinates[0])/179.0f),1);
	
	float thetaHraw = coordinates[2];
	float thetaDraw = coordinates[1];
	float phiDraw = coordinates[0];
	float3 infocolor = float3(0.0,0.0,0.0);
	if(thetaHraw*(2.0f/UNITY_PI)<0.1){infocolor[0]=1.0f;}
	if(thetaDraw*(2.0f/UNITY_PI)>0.8){infocolor[1]=1.0f;}
	if(thetaHraw*(2.0f/UNITY_PI)>0.8){infocolor[2]=1.0f;}
	
	
	return float3(infocolor*light.color);//phiDraw/180.0f,1);
	}



//Added by Jamie
half4 BRDF4_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0))
{
	
	if(_plot_mode>0.0f){
		//return half4(tex_xy[0],0,tex_xy[1],1);
		normal = float3(0,0,1);
		float halfTheta = (1-tex_xy[0])*UNITY_PI/2.0f;
		float halfDiff = (1-tex_xy[1])*UNITY_PI/2.0f;
		//float3 halfDir = float3(sin(halfTheta),0,cos(halfTheta));
		light.dir = normalize(float3(sin(halfDiff),0,cos(halfDiff)));
		viewDir = normalize(float3(-sin(halfDiff),0,cos(halfDiff)));
		normal = normalize(float3(0,sin(halfTheta),cos(halfTheta)));
		light.color = float3(0.5,0.5,0.5);
	}
	
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);
	
	half nl = saturate(dot(normal, light.dir));
    float nh = saturate(dot(normal, halfDir));
    half nv = saturate(dot(normal, viewDir));
    float lh = saturate(dot(light.dir, halfDir));
	float vh = saturate(dot(viewDir,halfDir));
	
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 uvec;
	float3 vvec;
	float3 local_normal = float3(0,0,1);
	if(nz<0.0f)
	{
		float a = 1.0f / (1.0f - nz);
		float b = nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, -b, nx);
		vvec = float3(b, ny * ny*a - 1.0f, -ny);
	}
	else{
		float a = 1.0f / (1.0f + nz);
		float b = -nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, b, -nx);
		vvec = float3(b, 1.0f - ny * ny * a, -ny);
	}
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
//	float3 transformed_normal = Unity_SafeNormalize(mul(Tmatrix,normal));
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	nl = dot(float3(0,0,1), local_light_dir);
	nv = dot(float3(0,0,1), local_view_dir);
	
    //half perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    //half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);



    diffColor = half3(_dr,_dg,_db);
	specColor = half3(_sr,_sg,_sb);
	float eta = _FresnelEta;
    float a = _RoughnessAlpha;
	float d;
	if(_mitsuba_dist_mode>0.0){d = CTTermMitsuba(local_half_dir,a);}
	else{d = CTTerm(nh,a);}
	//float g_smaller = min(nv,nl);
	//float g = min(1.0f,2.0*nh*g_smaller/vh);
	float g = mitsuba_G_Beckmann(local_light_dir,local_view_dir,local_half_dir,a);
	float f = mitsuba_Fresnel(vh,eta);
    float specularTerm = d*g*f;//a2 / (max(0.1f, lh*lh) * (roughness + 0.5f) * (d * d) * 4);


    // on mobiles (where half actually means something) denominator have risk of overflow
    // clamp below was added specifically to "fix" that, but dx compiler (we convert bytecode to metal/gles)
    // sees that specularTerm have only non-negative terms, so it skips max(0,..) in clamp (leaving only min(100,...))
#if defined (SHADER_API_MOBILE)
    specularTerm = specularTerm - 1e-4f;
#endif



#if defined (SHADER_API_MOBILE)
    specularTerm = clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
#endif
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = 0.0;
#endif


    half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));
    half3 color =   ((1.0f/3.14159f)*diffColor + specularTerm * specColor) * light.color * nl;
                    //+ gi.diffuse * diffColor
                    //+ surfaceReduction * gi.specular * specColor * f;
	if(_plot_mode>0.0f){return float4(pow(((1.0/3.14159)*diffColor + /*(1.0/3.14159)*/ specularTerm* specColor),1),1);}
	//if(_show_info_shader){color = color + INFORMATIVE_SHADER(diffColor, specColor, oneMinusReflectivity, smoothness, normal, viewDir, light, gi, F0, tex_xy);}
    return half4(pow(color, 1.0), 1);
}

/* GGX Research BRDF*/

inline float GGXTermResearch (float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float d = (NdotH * a2 - NdotH) * NdotH + 1.0f; // 2 mad
    return a2 / (3.14159f * d * d); //Added pi to match Mitsuba
                                    
}

inline float GGXTermMitsuba(float3 m, float roughness, float anisotropy = 1.0f){
	if(dot(m, float3(0.0f,0.0f,1.0f)) <= 0.0f){return 0.0f;}
	float cosTheta = dot(m,float3(0.0,0.0,1.0));//added by me to save a dot product computation
	//return cosTheta;
	float cosTheta2 = cosTheta*cosTheta;
	float m_alphaU = roughness;
	float m_alphaV = roughness*anisotropy;
	float beckmannExponent = ((m[0]*m[0]) / (m_alphaU * m_alphaU) + (m[1]*m[1]) / (m_alphaV * m_alphaV)) / cosTheta2;
	float root = (1.0f + beckmannExponent) * cosTheta2;
    float result = 1.0f / (3.14159f * m_alphaU * m_alphaV * root * root);
	if (result * cosTheta < 1e-20f){result = 0.0f;}
	return result;
}

float3 displayscale(float3 x, int i=0){
	return float3(-x[i],0,x[i]);
	return (x+float3(1,1,1))/2.0;
}

/*Simple Multiple Scattering*/
int E_TABLE_SIZE = 100;

float _ta;
float _tb;

float getTableEntry(float2 uv){
	//checked all combinations, this proved to be the only one that looked plausible
	return tex2D(_Etable,float2(1-uv[1],uv[0]));
}


float3 computeIndex(float v){
      v = 1.0f - v;
      if (v < 0){return float3(-1,-1,-1);}
      //v -= MIN_E_TABLE_V;
	  if (v < 0){v = 0;}
      float indexF = v * E_TABLE_SIZE;
      float min_index = floor(indexF);
      if (min_index < 0){min_index = 0;}
      float max_index = min_index + 1;
      if (max_index > E_TABLE_SIZE -1) max_index = E_TABLE_SIZE-1;
      if (indexF < min_index) return float3(-1,-1,-1);
      if (indexF > max_index) return float3(-1,-1,-1);
      float interp = indexF - min_index;
      return float3(min_index,max_index,interp);
    }

float getEfetch(float n_dot_v, float alpha){
	return getTableEntry(float2(n_dot_v, alpha));
}

float getE(float n_dot_v, float alpha){
      //float x_min, x_max, y_min, y_max;
      //float x_interp,  y_interp;
      /*bool validX = */float3 xp = computeIndex(alpha);
      /*bool validY = */float3 yp = computeIndex(n_dot_v);
      //if (!validX || !validY) return 0;
	  float x_min = xp[0];
	  float x_max = xp[1];
	  float x_interp = xp[2];
	  float y_min = yp[0];
	  float y_max = yp[1];
	  float y_interp = yp[2];
      float eXminYmin = getTableEntry(float2(x_min, y_min));
      float eXminYmax = getTableEntry(float2(x_min, y_max));
      float eXmaxYmin = getTableEntry(float2(x_max, y_min));
      float eXmaxYmax = getTableEntry(float2(x_max, y_max));
      float eYminXinterp = lerp(eXminYmin, eXmaxYmin, x_interp);
      float eYmaxXinterp = lerp(eXminYmax, eXmaxYmax, x_interp);
      float e1 = lerp(eYminXinterp, eYmaxXinterp, y_interp);
      float eXminYinterp = lerp(eXminYmin, eXminYmax, y_interp);
      float eXmaxYinterp = lerp(eXmaxYmax, eXmaxYmax, y_interp);
      float e2 = lerp(eXminYinterp, eXmaxYinterp, x_interp);
      return (e1+e2) * .5;
    }

 float computeMSE(float n_dot_v, float alpha){
      float palpha = alpha;
      float e = getEfetch(n_dot_v, palpha);//getE(n_dot_v, palpha);
      if (e > 0 && e < 1.0f) return (1.0f-e)/e;
      return 0.0f;
    }
	
//{
/*MSVG Based on Feng's Mitsuba Code*/

//Actually from mitsuba
inline int rel_eq(float x, float y, float thresh = 1e-5) { if(abs(x-y) < thresh) return 1; return 0; }

inline int rel_eq(float3 v1, float3 v2, float thresh = 1e-3) {
    if(rel_eq(v1.x, v2.x, thresh) + rel_eq(v1.y, v2.y, thresh) + rel_eq(v1.z, v2.z, thresh) == 3) return 1; return 0; 
}

inline int isSame(float x, float y) { return rel_eq(x, y); }
inline int isSame(float3 v1, float3 v2) {return rel_eq(v1, v2); }


struct Feng_Transform{
	float3x3 T;
	float3x3 invT;
};

float computeThetaForPhi0(float3 w) {
    float cosTheta = w.z;
    float cosPhi = 1;
    float sinPhi = 0; 
    float sinTheta = w.x/cosPhi;
    float theta = atan2(sinTheta, cosTheta);
    return theta;
}

Feng_Transform MyRotateY(float theta) {
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float3x3 m = float3x3(cosTheta, 0, sinTheta, 0, 1, 0, -sinTheta, 0, cosTheta);				
	float3x3 im = float3x3(cosTheta,0,-sinTheta,0,1,0,sinTheta,0,cosTheta);
    Feng_Transform trans;
	trans.T=m;
	trans.invT=im;
	return trans;
}

Feng_Transform MyRotateZ(float theta) {
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float3x3 m = float3x3(cosTheta, -sinTheta, 0, sinTheta, cosTheta, 0, 0, 0, 1);
    float3x3 im = float3x3(cosTheta,sinTheta,0,-sinTheta,cosTheta,0,0,0,1);
    Feng_Transform trans;
	trans.T=m;
	trans.invT=im;
	return trans;
}
 
struct VGFrame {
	void createRotationZ(float3 wh) {
        float t = atan2(wh.y, wh.x);
        /*
        if (t < 0) {
            t += Pi * 2.0;
        }
        */
        frameTheta = -t;
        w2l = MyRotateZ(frameTheta);
        l2w = MyRotateZ(-frameTheta);
    }

	void createRotationY(float3 wh) {
        float t = atan2(wh.x, wh.z);
        /*
        if (t < 0) {
            t += Pi * 2.0;
        }
        */
        frameTheta = -t;
        w2l = MyRotateY(frameTheta);
        l2w = MyRotateY(-frameTheta);
    }
float3 worldToLocal(float3 wW){
        return mul(w2l.T, wW);
    }
	
float3 localToWorld(float3 wl){
        //this is a bit of hack to take advantage of rotation being symmetric
        float3 t = wl;
        if (flipped>0) {
            t.x = -t.x;
            //N.y = -N.y;
        }
        return mul(l2w.T,t);
    }
    Feng_Transform w2l;
	Feng_Transform l2w;
    float frameTheta;
    int flipped;
};


	
struct EvalFrame : VGFrame {
    //EvalFrame() {flipped = false; };
	
	void createEvalFrame(float3 towo, float3 towi, int autoFlip) {
		owo = towo;
		owi = towi;
		
        wo = normalize(owo);
        wi = normalize(owi);

        owh = normalize((wo+wi));
        
        createRotationZ(owh);
        wo = worldToLocal(wo);
        wi = worldToLocal(wi);
        wh = worldToLocal(owh); 

        //CHECK(rel_eq(wo.y, -wi.y));

        wop = float3(wo.x, 0, wo.z);
        wop = normalize(wop);
        wip = float3(wi.x, 0, wi.z);
        wip = normalize(wip);
        theta_o = computeThetaForPhi0(wop);
        theta_i = computeThetaForPhi0(wip);
        flipped = 0;

        //y value and x value needs to be flipped 
        if (theta_o < 0 && autoFlip > 0) {
            theta_o *= -1;
            theta_i *= -1;
            wo.x *= -1;
            wi.x *= -1;
            wop.x *= -1;
            wip.x *= -1;
            //wo.y *= -1;
            //wi.y *= -1;
            flipped = 0;
        }
    }
        bool isSame(EvalFrame other) {
		return rel_eq(theta_o, other.theta_o) && rel_eq(theta_i, other.theta_i) 
			   //&& rel_eq(wo, other.wo) && rel_eq(wi, other.wi) 
			   && rel_eq(wop, other.wop) 
               && rel_eq(wip, other.wip) && rel_eq(owo, other.owo) 
               && rel_eq(owi, other.owi);
			  // && rel_eq(owh, other.owh);
    }
    
	//VGFrame frame;
    float3 wo, wi, wh, wop, wip, owo, owi, owh;
    float theta_o, theta_i;
};


struct Jacobian: VGFrame{
    void createJacobian(float3 towo, float3 towi, float3 towh, float teta){
		owo = towo;
		owi = towi;
		owh = towh;
		eta = teta;

        //float3 owhp = float3(-wh.x, 0, wh.z);
        //wih = vec3.dot(wi, wh)

        createRotationY(owh);

        N = float3(0, 0, 1);
        Ng = worldToLocal(N);
        H = worldToLocal(owh);
		//H = Unity_SafeNormalize(H);//Added by Jamie
        HP = reflect(H, Ng);
		//HP = Unity_SafeNormalize(HP);//added by Jamie

        //CHECK(rel_eq(H, float3(0, 0, 1)));
        //assert(vec3.Equal(H, vec3.Vec3(0, 0, 1)))
        //assert(vec3.Equal(self.HP, self.rotateH.rotate(vec3.Vec3(-wh.x, 0, wh.z))))

        wo = worldToLocal(owo);
        wi = worldToLocal(owi);
        //assert(vec3.EqualValue(wi.z, wih))

        dxpdxa = - 1.0 + 2.0 * Ng.x * Ng.x;
        dypdya = - 1.0 + 2.0 * Ng.y * Ng.y;
        dzpdxa = 2.0 * Ng.x * Ng.z;
    }
    
    float3 getH(int bounce) {
        return (bounce % 2) ? H: HP;
    }

	//manually encode the recursion
    
	
	float2 computeDxaDya1(int bounce, inout float3 w, inout float3 F) {
        if (bounce == 0) {
            w = -wo;
            F = float3(1,1,1);
            return float2(0, 0);    
        } else {
            float3 wp;
            float2 pDxDy = computeDxaDya2(bounce - 1, wp, F);
            float pDxdxa = pDxDy.x;
            float pDydya = pDxDy.y;

            float3 h = getH(bounce);
            float kp = dot(wp, h);
            w = reflect(-wp, h);
			w = Unity_SafeNormalize(w);
            //if (true) F *= mitsuba_Fresnel(dot(w, h), eta);//changed by Jamie
			F = F * mitsuba_Fresnel(dot(w, h), eta);//changed by Jamie
            float dxdxa = 0, dydya = 0;
            if (bounce % 2 == 0) {
                float pDzdxa = -wp.x/wp.z * pDxdxa;
                dxdxa = pDxdxa - 2.0 * kp * dxpdxa;
                dxdxa -= 2.0 * HP.x * (pDxdxa * HP.x + wp.x * dxpdxa + pDzdxa * HP.z + wp.z * dzpdxa);
                dydya = pDydya + 2.0 * kp;
            } else {
                dxdxa = pDxdxa - 2.0 * kp;
                dydya = pDydya - 2.0 * kp;
            }
            return float2(dxdxa, dydya);
        }
    }
	
	float2 computeDxaDya2(int bounce, inout float3 w, inout float3 F) {
        if (bounce == 0) {
            w = -wo;
            F = float3(1,1,1);
            return float2(0, 0);    
        } else {
            float3 wp;
            float2 pDxDy = computeDxaDya3(bounce - 1, wp, F);
            float pDxdxa = pDxDy.x;
            float pDydya = pDxDy.y;

            float3 h = getH(bounce);
            float kp = dot(wp, h);
            w = reflect(-wp, h);
			w = Unity_SafeNormalize(w);
            F = F * mitsuba_Fresnel(dot(w, h), eta);//changed by Jamie
            float dxdxa = 0, dydya = 0;
            if (bounce % 2 == 0) {
                float pDzdxa = -wp.x/wp.z * pDxdxa;
                dxdxa = pDxdxa - 2.0 * kp * dxpdxa;
                dxdxa -= 2.0 * HP.x * (pDxdxa * HP.x + wp.x * dxpdxa + pDzdxa * HP.z + wp.z * dzpdxa);
                dydya = pDydya + 2.0 * kp;
            } else {
                dxdxa = pDxdxa - 2.0 * kp;
                dydya = pDydya - 2.0 * kp;
            }
            return float2(dxdxa, dydya);
        }
    }
	
	float2 computeDxaDya3(int bounce, inout float3 w, inout float3 F) {
        if (bounce == 0) {
            w = -wo;
            F = float3(1,1,1);
            return float2(0, 0);    
        } else {
            float3 wp;
            float2 pDxDy = computeDxaDya4(bounce - 1, wp, F);
            float pDxdxa = pDxDy.x;
            float pDydya = pDxDy.y;

            float3 h = getH(bounce);
            float kp = dot(wp, h);
            w = reflect(-wp, h);
			w = Unity_SafeNormalize(w);
            F = F * mitsuba_Fresnel(dot(w, h), eta);//changed by Jamie
            float dxdxa = 0, dydya = 0;
            if (bounce % 2 == 0) {
                float pDzdxa = -wp.x/wp.z * pDxdxa;
                dxdxa = pDxdxa - 2.0 * kp * dxpdxa;
                dxdxa -= 2.0 * HP.x * (pDxdxa * HP.x + wp.x * dxpdxa + pDzdxa * HP.z + wp.z * dzpdxa);
                dydya = pDydya + 2.0 * kp;
            } else {
                dxdxa = pDxdxa - 2.0 * kp;
                dydya = pDydya - 2.0 * kp;
            }
            return float2(dxdxa, dydya);
        }
    }
	
	float2 computeDxaDya4(int bounce, inout float3 w, inout float3 F) {
        if (bounce == 0) {
            w = -wo;
            F = float3(1,1,1);
            return float2(0, 0);    
        } else {
            float3 wp;
            float2 pDxDy = computeDxaDya5(bounce - 1, wp, F);
            float pDxdxa = pDxDy.x;
            float pDydya = pDxDy.y;

            float3 h = getH(bounce);
            float kp = dot(wp, h);
            w = reflect(-wp, h);
			w = Unity_SafeNormalize(w);
            F = F * mitsuba_Fresnel(dot(w, h), eta);//changed by Jamie
            float dxdxa = 0, dydya = 0;
            if (bounce % 2 == 0) {
                float pDzdxa = -wp.x/wp.z * pDxdxa;
                dxdxa = pDxdxa - 2.0 * kp * dxpdxa;
                dxdxa -= 2.0 * HP.x * (pDxdxa * HP.x + wp.x * dxpdxa + pDzdxa * HP.z + wp.z * dzpdxa);
                dydya = pDydya + 2.0 * kp;
            } else {
                dxdxa = pDxdxa - 2.0 * kp;
                dydya = pDydya - 2.0 * kp;
            }
            return float2(dxdxa, dydya);
        }
    }
	
	float2 computeDxaDya5(int bounce, inout float3 w, inout float3 F) {
        if (bounce == 0) {
            w = -wo;
            F = float3(1,1,1);
            return float2(0, 0);    
        } else {
            float3 wp;
            float2 pDxDy = computeDxaDya6(bounce - 1, wp, F);
            float pDxdxa = pDxDy.x;
            float pDydya = pDxDy.y;

            float3 h = getH(bounce);
            float kp = dot(wp, h);
            w = reflect(-wp, h);
			w = Unity_SafeNormalize(w);
            F = F * mitsuba_Fresnel(dot(w, h), eta);//changed by Jamie
            float dxdxa = 0, dydya = 0;
            if (bounce % 2 == 0) {
                float pDzdxa = -wp.x/wp.z * pDxdxa;
                dxdxa = pDxdxa - 2.0 * kp * dxpdxa;
                dxdxa -= 2.0 * HP.x * (pDxdxa * HP.x + wp.x * dxpdxa + pDzdxa * HP.z + wp.z * dzpdxa);
                dydya = pDydya + 2.0 * kp;
            } else {
                dxdxa = pDxdxa - 2.0 * kp;
                dydya = pDydya - 2.0 * kp;
            }
            return float2(dxdxa, dydya);
        }
    }
	
	
	float2 computeDxaDya6(int bounce, inout float3 w, inout float3 F) {
            w = -wo;
			w = Unity_SafeNormalize(w);
            F = float3(1,1,1);
            return float2(0, 0);
    }
	 
	
	
	

    float computeJacobian(int bounce, inout float3 F) {
        float3 wr;
        float2 dxy = computeDxaDya1(bounce, wr, F);

        //if (!rel_eq(wr, wi)) {
        //    float2 dxy = computeDxaDya1(bounce, wr, F);
            //std::cout << "computed outgoing direction: " << wr <<"\n";
            //std::cout << "expected outgoing direction: " << wi <<"\n";
            //fflush(stdout);
        //}
        float denom = abs(dxy.x * dxy.y);
        if (denom < 1e-6) return 0.0f;
        float nom = abs(wi.z);
        float jacobian = nom/denom;
        return jacobian;
    }

    float eta;
    float3 N, Ng, H, HP, wo, wi, owo, owi, owh;
    float dxpdxa, dypdya, dzpdxa;

};
    
float3 computeZipinNormal(float thetaM, int side, float3 wop) {
    float3 n = float3(cos(thetaM), 0, sin(thetaM));
    //n.x *= wop.x >= 0 ? 1: -1;
    if (side == -1 ){
        n.x *= -1;
    }
    return n;
}

float AbsCosTheta(float3 v) { return abs(v.z);}

struct Hit {
    void createHit(float tthetaR, int tbounce, int tside, float tGFactor) {
		thetaR = tthetaR;
		bounce = tbounce;
		side = tside;
		GFactor = tGFactor;
	}
	
	void createHit() {
		thetaR = 0.0f;
		bounce = 0;
		side = -1;
		GFactor = 0.0f;
	}

    bool isHit(float thetaI, int bcount, int s) {
        return abs(thetaR-thetaI) < 1e-6 && bounce == bcount && side == s;
    }
    
    float thetaR;
    int bounce;
    int side;
	float GFactor;
    //xmin, xmax, ratio are only used for debugging 
};

struct Hits{
	Hit hits[5];
};

class VGroove {
	float sumG;
	int maxBounce, minBounce;
	Hit theHits[5];
	int hitIndex;
	void makeVGroove(int tmaxBounce, int tminBounce){
		maxBounce=tmaxBounce;
		minBounce=tminBounce;
		sumG=0.0f;
		hitIndex = 0;
	}
	
	void clear(){
		sumG = 0;
		for(int i = 0;i<5;i++){
			Hit newHit;
			newHit.createHit();
			theHits[i] = newHit;
		}
		hitIndex = 0;
	}
		
	bool computeRightThetaM(float thetaO, float thetaI, int n, inout float theta) {
		//added so theta has a value
		theta = 0.0f;
		//return true;
		float xchi = -thetaI;
		if ((n+1)%2 == 1) xchi *= -1;
		theta = (UNITY_PI + thetaO - xchi) *.5f/n;
		if (theta > thetaO && theta < .5f * UNITY_PI + 1e-6) return true;
		return false; 
	}
	
	bool computeLeftThetaM(float thetaO, float thetaI, int n, inout float theta) {
		//added so theta has a value
		theta = 0.0f;
		//return true;
		float xchi = -thetaI;
		if (n%2 == 1) xchi *= -1;
		theta = (UNITY_PI - thetaO - xchi) *.5/n;
		if (theta > 1e-6 && theta < .5 * UNITY_PI + 1e-6) return true;
		return false;
	} 

	bool computeThetaM(float thetaO, float thetaI, int bounceCount, int side, inout float thetaM) {
		if (side == 1)  {
			return computeLeftThetaM(thetaO, thetaI, bounceCount, thetaM);
		} else {
			return computeRightThetaM(thetaO, thetaI, bounceCount, thetaM);
		}
	}
	
	void writeHit(Hit hit, int i){
		if(i==0){theHits[0]=hit;}
		if(i==1){theHits[1]=hit;}
		if(i==2){theHits[2]=hit;}
		if(i==3){theHits[3]=hit;}
		if(i==4){theHits[4]=hit;}
		
	}
	
	void addHit(float xi, int bcount, int side, float GFactor) {
   
		//if (bcount >= minBounce && bcount <=maxBounce) { 
		//	if (GFactor > 1e-5 ) {
				GFactor = min(1.0f, GFactor);
				Hit newHit;
				newHit.createHit(-xi, bcount, side, GFactor);
				writeHit(newHit,hitIndex);// This must be unrolled by hand
				hitIndex = hitIndex+1;
				sumG += GFactor;
		//	}
		//}
	}
	
	inline float stableCeiling(float psi, float gAngle) {
    
		int k = (int)(psi/gAngle);
    //CHECK(k >= 0);
		if (psi - k * gAngle < 1e-6) return k;
		return k + 1;
	}
	
	void leftEval(float theta, float phi, float maxX, bool hasNear, float hitrange) {

		float gAngle = 2.0 * theta;
		float psi_min = UNITY_PI - 2.0 *(theta + phi);

		float sinThetaPhi = sin(theta+phi);

		float psi_max = UNITY_PI - (theta+phi);

		float zMax = 1;

		if (hasNear == false && sinThetaPhi > 0) {
			//this is exactly Gi because when all the rays go to the left side
			//the max left side visible is Gi
			//equation 9 in Zipin Paper, hitrange is 2sinTheta

			zMax = hitrange * cos(phi)/ sinThetaPhi;
			psi_max -= asin((1.0-zMax) * sinThetaPhi);
		} 

		//print(mt.degrees(psi_min), mt.degrees(psi_max))

		int n_min = stableCeiling(psi_min, gAngle);
		int n_max = stableCeiling(psi_max, gAngle);
		
		//if n_max - n_min > 1:
		//    print(theta, phi, n_max, n_min) 

		float xi_min = UNITY_PI - phi - n_min * gAngle;
		float xi_max = UNITY_PI - phi - n_max * gAngle;
		
		float x_min_intersect = -hitrange * .5;
		float x_max_intersect = maxX;
		float xrange = maxX - x_min_intersect;

		if (n_min%2) xi_min *= -1;
		if (n_max%2) xi_max *= -1;

		if (n_max > n_min) {
			//compute z critical intersect, length from circle center
			int k = n_min;
			float criticalAngle = UNITY_PI - (theta+phi) - k* gAngle;
			float z = 0;
			if (criticalAngle > 0) {
				z = sin(criticalAngle)/sinThetaPhi;
			}
			addHit(xi_max, n_max, 1, zMax - (1-z));
			addHit(xi_min, n_min, 1, 1-z);
		
			//Float x_critical_intersect = x_min_intersect + (1-z) / zMax  * xrange
			//addHit(hits, xi_max, n_max, 'l', zMax - (1-z), (x_critical_intersect, x_max_intersect), hitrange);
			//addHit(hits, xi_min, n_min, 'l', 1-z, (x_min_intersect, x_critical_intersect), hitrange);
		} else {
			addHit(xi_min, n_min, 1, zMax);

			//addHit(hits, xi_min, n_min, 'l', zMax, (x_min_intersect, x_max_intersect), hitrange);
		}
	}
	
	float rightEval(float theta, float phi, float hitrange) {

		float gAngle = 2.0 * theta;
	   
		//Feng's correction to near psi computation from Paper 
		float psi_max = UNITY_PI -(theta-phi);
		float psi_min = UNITY_PI - 2.0 * (theta-phi);

		//#Zipin Paper near psi computation 
		//psi_max = mt.pi - 2.0 * phi
		//psi_min = mt.pi - theta - phi

		float x_near_min = cos(theta) * tan(phi);

		int n_min = stableCeiling(psi_min, gAngle);
		int n_max = stableCeiling(psi_max, gAngle);

		//if (n_max - n_min) > 1:
		//    print(theta, phi, n_max, n_min) 

		float xi_min = UNITY_PI + phi - n_min * gAngle;
		float xi_max = UNITY_PI + phi - n_max * gAngle;

		if (n_min%2 == 0) xi_min *= -1;
		if (n_max%2 == 0) xi_max *= -1;
		
		float x_min_intersect = x_near_min;
		float x_max_intersect = sin(theta);

		if (n_min == n_max || theta - phi < 1e-5) {
			//addHit(hits, xi_min, n_min, 'right', 1.0, (x_min_intersect, x_max_intersect), hitrange)
			addHit(xi_min, n_min, -1, 1.0);
		} else {
			int k = n_min;
			float criticalAngle = UNITY_PI - (theta-phi) -  gAngle * k;
			float z = 0;
			
			if (criticalAngle > 0) {
				z = sin(criticalAngle) /sin(theta-phi);
			}
			addHit(xi_min, n_min, -1, 1.0-z);
			addHit(xi_max, n_max, -1, z);
			//x_critical_intersect = x_min_intersect + (x_max_intersect - x_min_intersect) * z;
			//addHit(hits, xi_min, n_min, 'right', 1.0-z, (x_critical_intersect, x_max_intersect), hitrange)
			//addHit(hits, xi_max, n_max, 'right', z, (x_min_intersect, x_critical_intersect), hitrange)
		}
		return x_near_min;
	}
	
	
	/*These should actually return an array of hits*/
	Hits completeEval(float thetaR, float phiR) {
		clear();
		float xmax = sin(thetaR);
		if (phiR > thetaR) {
			leftEval(thetaR, phiR, xmax, false, xmax*2);
		} else {
			float  x_near_min = rightEval(thetaR, phiR, xmax*2); 
			leftEval(thetaR, phiR, x_near_min, true,  xmax*2);
		}
		Hits hitstruct;
		hitstruct.hits = theHits;
		return hitstruct;
	}

	Hits leftEvalOnly(float thetaR, float phiR) {
		clear();
		bool hasNear = (thetaR > phiR);
		float xmax = sin(thetaR);
		float farMax = hasNear? cos(thetaR) * tan(phiR) : xmax;
		leftEval(thetaR, phiR, xmax, hasNear, xmax*2);
		Hits hitstruct;
		hitstruct.hits = theHits;
		return hitstruct;
	}

	Hits rightEvalOnly(float thetaR, float phiR) {
		clear();
		float xmax = sin(thetaR);
		float  x_near_min = rightEval(thetaR, phiR, xmax*2); 
		Hits hitstruct;
		hitstruct.hits = theHits;
		return hitstruct;
	}
	
	
	float inverseEval(float thetaO, float thetaI, int bounceCount, int side, inout float thetaM) { 
		
		float GFactor = 0;

		//CHECK(thetaO >= 0);
		if (thetaO + .0001 > .5 * UNITY_PI) return GFactor; 
		
		clear();
		if (side == -1) {
			bool validGroove = computeRightThetaM(thetaO, thetaI, bounceCount, thetaM);//fails here?
			if (validGroove) rightEvalOnly(thetaM, thetaO);
		} else {
			bool validGroove = computeLeftThetaM(thetaO, thetaI, bounceCount, thetaM);//fails here?
			if (validGroove) leftEvalOnly(thetaM, thetaO);
		}
		for (int i = 0; i < hitIndex; i++) {
			if (theHits[i].isHit(thetaI, bounceCount, side)) {
				GFactor = theHits[i].GFactor;
				return GFactor;
			}
		}
		return 0.0f; 
	}
	
	
	
	/*
  public:
    VGroove(int maxBounce, int minBounce):maxBounce(maxBounce), minBounce(minBounce), sumG(0){}
    void clear() { sumG = 0; theHits.clear(); } DONE
    inline std::vector<Hit>& completeEval(Float thetaG, Float thetaO); DONE
    inline std::vector<Hit>& leftEvalOnly(Float thetaG, Float thetaO); DONE
    inline std::vector<Hit>& rightEvalOnly(Float thetaG, Float thetaO); DONE

    inline Float inverseEval(Float thetaO, Float thetaI, int bounceCount, char side, Float& thetaM); DONE
    std::vector<Hit> theHits;
    const int maxBounce, minBounce;
    Float sumG;

  private:
    inline void leftEval(Float theta, Float phi, Float maxX, bool hasNear, Float hitrange); DONE
    inline Float rightEval(Float theta, Float phi, Float hitrange); DONE
    inline void addHit(Float xi, int bcount, char side, Float GFactor); DONE
    inline static bool computeThetaM(Float thetaO, Float thetaI, int bounceCount, char side, Float& thetaM); DONE
    inline static bool computeRightThetaM(Float thetaO, Float thetaI, int bounceCount, Float& theta); DONE
    inline static bool computeLeftThetaM(Float thetaO, Float thetaI, int bounceCount, Float& theta); DONE
	*/
	
};

bool SameHemisphere(float3 v1, float3 v2) { return v1.z * v2.z > 0.0f; }


class VGrooveReflection {
	
	float microfacetReflectionWithoutG(float3 wo, float3 wi, float3 wh, float alpha) {
		float cosThetaO = AbsCosTheta(wo);
		float cosThetaI = AbsCosTheta(wi);
		// Handle degenerate cases for microfacet reflection
		if (cosThetaI == 0 || cosThetaO == 0) return 0;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return 0;
		//Spectrum F = fresnel->Evaluate(dot(wi, wh));
		return GGXTermResearch (dot(wh,float3(0,0,1)), alpha) / (4 * cosThetaI * cosThetaO);
	}
	
	float computeGFactor(EvalFrame evalFrame, inout VGroove vgroove, int bounce, int side, inout float3 wm) {
		//added so that wm must have a value
		wm = float3(0,0,0);
		//debug single bounce
		
		float thetaM = 0;
		float GFactor = vgroove.inverseEval(evalFrame.theta_o, evalFrame.theta_i, bounce, side, thetaM);//fails on this function call!
		
		if (GFactor > 0) {
			GFactor = min(1.0f, GFactor);
			wm = computeZipinNormal(thetaM, side, evalFrame.wop);
			//if (bounce == 1) {
				//no relation between frameTheta and thetaM (frameTheta is related to phi_h not theta_h
				//float3 wh = normalize((evalFrame.wo + evalFrame.wi));
				//float mh = dot(wm, wh);
				//if (!rel_eq(mh, 1.f)){
				//    std::cout<<"wm: "<< wm << " wh: "<<wh<<"\n";
				//    fflush(stdout);
				//}
			//}
		}
		return GFactor;
		}
	
	
	float computeBounceBrdf(EvalFrame evalFrame, VGroove vgroove, int bounce, int side, inout float pdf, inout float3 F){
		//F = float3(1,1,1);//moved up for debug
		float3 wm;
		float GFactor = computeGFactor(evalFrame, vgroove, bounce, side, wm);//fails in this function call

		pdf = 0.0f;
		
		float brdf = 0.0f;
		if (GFactor > 0) {
			float wom = dot(evalFrame.wo, wm);
			if (wom > 1e-6f) {
				float3 owm = evalFrame.localToWorld(wm);
				float val = microfacetReflectionWithoutG(evalFrame.owo, evalFrame.owi, owm, alpha);
				//float mpdf = microfacetPdf(evalFrame.owo, owm);

				Jacobian jacobian;
				jacobian.createJacobian(evalFrame.wo, evalFrame.wi, wm, eta);
				float Jac = jacobian.computeJacobian(bounce, F);
				if (Jac > 1e-6) {
					float J = Jac * wom * 4; 
					brdf = val * J * GFactor;
					//pdf = mpdf * J * GFactor / vgroove.sumG;
					return brdf;
					
				}
			}
		}
		return brdf;
	}
	
	float3 eval(EvalFrame evalFrame, float3 wo, float3 wi, inout float pdf) {
		
		//pdf = .5/Pi;
		//return MicrofacetReflection::f(wo, wi);

		float3 brdf = float3(0,0,0);
		pdf = 0;
		if (!SameHemisphere(wo, wi)) return brdf;
		//if (evalFrame.theta_o < 1e-6) return brdf;
		
		VGroove vgroove;
		vgroove.makeVGroove(maxBounce, minBounce);
		
		for (int n = minBounce; n<=maxBounce; n++) {//fails in this loop!
			float tpdf = 0; 
			float3 F = float3(0,0,0);
			float tbrdf = computeBounceBrdf(evalFrame, vgroove, n, -1, tpdf, F);//fails in this function call
			brdf += tbrdf * F;
			
			pdf += tpdf;
			if (n == 1 && tbrdf > 0) continue;
			//float3 F2;
			tbrdf = computeBounceBrdf(evalFrame, vgroove, n, 1, tpdf, F);
			brdf += tbrdf * F;
			pdf += tpdf;
		}
		return R*brdf;
	}
	
	void createVGrooveReflection(float3 tR, float talpha, float teta, int tmaxBounce, int tminBounce, bool tuniSample){
		R = tR;
		alpha = talpha;
		eta = teta;
		maxBounce = tmaxBounce;
		minBounce = tminBounce;
		uniSample = tuniSample;
	}
	
	float3 f(float3 wo, float3 wi, inout float pdf) {
		
		float tmppdf = 0.0f;
		EvalFrame evalFrame;
		evalFrame.createEvalFrame(wo, wi, 0); //CHECK THIS LAST PARAMETER
		
		float3 brdf =  eval(evalFrame, wo, wi, tmppdf);//fails here!
		
		pdf = uniSample? 0.5f/UNITY_PI : tmppdf;
		return brdf;
	}
	
	float eta;
	float alpha;
	float3 R;
	int maxBounce, minBounce;
	bool uniSample;
	
	/*
  public:
    // MicrofacetReflection Public Methods
    VGrooveReflection(const Spectrum &R,
                      MicrofacetDistribution *distribution, const Spectrum& eta, const Spectrum& k, 
                      int maxBounce = 3, int minBounce = 1, bool uniSample = true);
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    Spectrum f(const Vector3f &wo, const Vector3f &wi, Float& pdf) const;
    
    Spectrum Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u,
                      Float *pdf) const;
    Float Pdf(const Vector3f &wo, const Vector3f &wi) const;
    //std::string ToString() const;
    bool testPDF() const;

  private:

    //uniform sampling for testing
    Spectrum UniSample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u,
                      Float *pdf) const;

    Float microfacetReflectionWithoutG(const Vector3f& wo, const Vector3f& wi,
                   const Vector3f& wh) const;
    Float microfacetPdf(const Vector3f& wo, const Vector3f& wh) const;
    Float computeBounceBrdf(const EvalFrame& evalFrame, VGroove& vgroove, int bounce, char side,
                    Float& pdf, Spectrum& F) const; CALLED BY eval
    Spectrum eval(const EvalFrame& evalFrame, const Vector3f &wo, const Vector3f &wi, Float &pdf) const; NEEDED

    Float computeGFactor(const EvalFrame& evalFrame, VGroove& vgroove, int bounce, 
                         char side, Vector3f& wm) const;

    Float computePdfIntegral(Float thetaI) const;

    

	Spectrum R;
    MicrofacetDistribution *distribution;
	MetalFresnel fresnel;
    const int maxBounce, minBounce;
    bool uniSample;
	*/
};
float Frame_cosTheta(float3 v){
	return v[2];
}
	
	//eval method here
	float3 MSVG_evalWithPdf(float3 wi, float3 wo, float3 R, float alpha, float eta, int scatteringOrderMax, int scatteringOrderMin) {
        if (!SameHemisphere(wi, wo)){return float3(0,0,0);}
		
        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        
        VGrooveReflection vg;
		
		vg.createVGrooveReflection(R, alpha, eta, scatteringOrderMax, scatteringOrderMin, false);
		
		float pdf = 0.0f;
		if(wo.z<=0.0f){return float3(1,0,0);}
		return vg.f(wi, wo, pdf);// * Frame_cosTheta(wo);//wi and wo names do not match up
    }
	
//TODO: Literally everything for MSVG
//{
/*MSVG based on paper*/
/*Algorithm from 3.3*/ 
//s = 1 for l, -1 for r
float2 algorithm33(float thetaI, float thetaO, float n, float s){
	/*Step 1*/
	float thetaG;
	if(s>0.0){thetaG = (UNITY_PI - thetaI - pow(-1.0,n+1)*thetaO)/(2.0*n);}
	else{thetaG = (UNITY_PI + thetaI - pow(-1.0,n)*thetaO)/(2.0*n);}
	/*Step 2*/
	float psimin = UNITY_PI - 2*(thetaG + thetaI);
	/*Step 3*/
	float vmax;
	float psimax;
	if(thetaI>=thetaG){
		vmax = (2*sin(thetaG)*cos(thetaO))/(sin(thetaG+thetaI));
		psimax = UNITY_PI - (thetaG + thetaI) - asin((1.0f-vmax)*sin(thetaG + thetaI));
	}
	else{
		vmax = 1.0f;
		psimax = UNITY_PI - (thetaG + thetaI);
	}
	/*Step 4*/
	float nmin = ceil(psimin/(2.0f*thetaG));
	float nmax = ceil(psimax/(2.0f*thetaG));
	/*Step 5*/
	float vc = 1 - sin(UNITY_PI - (thetaG+thetaI) - 2.0*nmin*thetaG)/sin(thetaG+thetaI);
	/*Step 6*/
	float Gmin = vc;
	float Gmax = vmax - vc;
	/*Step 7*/
	if(nmin == n){
		return float2(thetaG,Gmin);
	}
	/*Step 8*/
	if(nmax == n){
		return float2(thetaG,Gmax);
	}
	/*Step 9*/
	return float2(0,0);
}

//Algorithm from 5.1

float4 algorithm51(float3 local_view_dir,float3 local_light_dir,float3 local_half_dir,float n, float s){
		/*Spin the local frame so that the half vector is on xz plane*/
		float3 yaxis = float3(0,1,0);
	
		float3 halfxy = Unity_SafeNormalize(local_half_dir * float3(1,1,0));
		float ang = dot(halfxy,yaxis);
		float3x3 Smatrix = float3x3(cos(ang),-sin(ang),0,sin(ang),cos(ang),0,0,0,1);
		float3x3 invSmatrix = float3x3(cos(-ang),-sin(-ang),0,sin(-ang),cos(-ang),0,0,0,1);
		local_light_dir = Unity_SafeNormalize(mul(Smatrix,local_light_dir));
		local_view_dir = Unity_SafeNormalize(mul(Smatrix,local_view_dir));
		local_half_dir = Unity_SafeNormalize(mul(Smatrix,local_half_dir));
		float flipx = 0;
		if(local_view_dir[0]<0.0f){
			flipx = 1.0f;
			local_view_dir = local_view_dir * float3(-1,1,1);
			local_light_dir = local_light_dir * float3(-1,1,1);
			local_half_dir = local_half_dir * float3(-1,1,1);
		}
	
	
	/*Step 3*/
	float3 wop = local_view_dir * float3(1,0,1);
	float3 wip = local_light_dir * float3(1,0,1);
	
	/*Step 4*/
	float thetaO = acos(wop[2]);
	float thetaI = acos(wip[2]);
	
	/*Step 5*/
	
	float2 alg33res = algorithm33(thetaO,thetaI,n,s);
	float thetaG = alg33res[0];
	float G_MSVG = alg33res[1];
	
	/*Step 6*/
	float3 wH = float3(cos(thetaG),0,sin(thetaG));
	if(wop[0]<0.0f){wH=wH*float3(-1,1,1);}
	
	/*Step 7*/
	if(s<0.0f){wH=wH*float3(-1,1,1);}
	
	/*Step 8*/
	if(flipx>0){
		wH = wH*float3(-1,1,1);
	}
	wH = Unity_SafeNormalize(mul(invSmatrix,wH));
	return float4(wH,G_MSVG);
}

float MSVG_D(float3 wH, float a){
	return GGXTermMitsuba(wH,a);
}

float MSVG_F(float3 wo, float3 wH, float n, float eta){
	float3 Rj = -1*wo;
	float3 zaxis = float3(0,0,1);
	float3 wHp = 2*dot(zaxis,wH)*zaxis - wH;
	float totalFresnel = 1.0f;
	for(int j=1; j<(int)n;j++){
		float3 Hj;
		if(j%2==0){Hj = wHp;}
		else{Hj = wH;}
		float3 Kj1 = Rj * Hj;
		Rj = Rj - 2*Kj1*Hj;
		float localF = mitsuba_Fresnel(dot(-1*Rj,Hj),eta);
		totalFresnel = totalFresnel*localF;
	}
	return 1.0f;
	return totalFresnel;
}

float MSVG_J(float3 wH, float3 wHp,float n){
	return 1.0f;
}

float MSVGfr1(float3 wo, float3 wi, float n, float s, float3 local_half_dir, float a, float eta){
	float4 alg51res = algorithm51(wo, wi, local_half_dir, n, s);
	float3 wH = float3(alg51res[0],alg51res[1],alg51res[2]);
	float G_MSVG = alg51res[3];
	return G_MSVG;
	return MSVG_F(wo,wH,n,eta)*MSVG_J(wH, wi, n) * G_MSVG * dot(wo,wH)*MSVG_D(wH, a);
}

float MSVGfr2(float3 wo, float3 wi, float n, float s, float3 local_half_dir, float a, float eta, float nv, float nl){
	//return 1.0f;
	return (MSVGfr1(wo,wi,n,1.0f,local_half_dir,a,eta)+MSVGfr1(wo,wi,n,-1.0f,local_half_dir,a,eta));//(nv*nl);
}
//}

inline float GVCavity1(float3 wo, float3 wh) {
        float denom = dot(wo, wh);
        if (denom < 1e-6) return 1;
        float nom = 2 * wo.z * wh.z;
        if (nom < 1e-6) return 0;

        float g = nom/denom;
        return min(g, 1.0f);
    }

inline float GGXVG_G(float3 wi, float3 wo, float3 m) {
			return min(GVCavity1(wi, m), GVCavity1(wo, m));
    }


float _use_MSVG;
float _GGXVG;
float _albedo_from_floats;

struct iHoldFloat {
	float c;
	void changeToOne(){c=1.0f;}
};
//}
//Added by Jamie


half4 BRDF5_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0))
{
	
	//return half4(1,1,1,1);
	if(_plot_mode>0.0f){
		//return half4(tex_xy[0],0,tex_xy[1],1);
		normal = float3(0,0,1);
		float halfTheta = (1-tex_xy[0])*UNITY_PI/2.0f;
		float halfDiff = (1-tex_xy[1])*UNITY_PI/2.0f;
		//float3 halfDir = float3(sin(halfTheta),0,cos(halfTheta));
		light.dir = normalize(float3(sin(halfDiff),0,cos(halfDiff)));
		viewDir = normalize(float3(-sin(halfDiff),0,cos(halfDiff)));
		normal = normalize(float3(0,sin(halfTheta),cos(halfTheta)));
		//light.color = float3(0.5,0.5,0.5);
	}
	//return float4(displayscale(normal,2),1);
	//normal = norm(normal);
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);
	//return float4(displayscale(halfDir,2),1);
    half nl = saturate(dot(normal, light.dir));
    float nh = saturate(dot(normal, halfDir));
    half nv = saturate(dot(normal, viewDir));
    float lh = saturate(dot(light.dir, halfDir));
	float vh = saturate(dot(viewDir,halfDir));
	
	/*Get local shading frame directions for view, light, and half vectors
	Algorithm taken from Duff et al. "Building an Orthonormal Basis, Revisited",
	Journal of Computer Graphics, Vol. 6, No. 1, 2017*/
	
	
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 uvec;
	float3 vvec;
	float3 local_normal = float3(0,0,1);
	if(nz<0.0f)
	{
		float a = 1.0f / (1.0f - nz);
		float b = nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, -b, nx);
		vvec = float3(b, ny * ny*a - 1.0f, -ny);
	}
	else{
		float a = 1.0f / (1.0f + nz);
		float b = -nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, b, -nx);
		vvec = float3(b, 1.0f - ny * ny * a, -ny);
	}
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	
//	float3 transformed_normal = Unity_SafeNormalize(mul(Tmatrix,normal));
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	

	
	nl = dot(float3(0,0,1), local_light_dir);
	nv = dot(float3(0,0,1), local_view_dir);
	
    //half perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    //half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
	
	//Core BRDF code starts here
	float a = smoothness;
	if(_albedo_from_floats){
	diffColor = half3(_dr,_dg,_db);
	specColor = half3(_sr,_sg,_sb);
	a = _RoughnessAlpha;
	}
	float3 floatspeccolor = float3(_sr,_sg,_sb);
	//return float4(floatspeccolor,1);
	float eta = _FresnelEta;
	float d;
	float g;
	float f;
	float monoSpecTerm = 0.0f;
	float3 specularTerm = float3(0,0,0);
	
	if(_use_MSVG==0.0f){
	
		if(_mitsuba_dist_mode>0.0){d = GGXTermMitsuba(local_half_dir,a);}
		else{d = GGXTermResearch(nh,a);}
		
		if(_GGXVG>0.0f){g = GGXVG_G(local_light_dir,local_view_dir,local_half_dir);}
		else{g = mitsuba_G(local_light_dir,local_view_dir,local_half_dir,a);}
		//float g_smaller = min(nv,nl);
		//float g = min(1.0f,2.0*nh*g_smaller/vh);
		f = mitsuba_Fresnel(vh,eta);
		//float f = FresnelTerm((eta-1.0f)*(eta-1.0f)/((eta+1.0f)*(eta+1.0f)),vh);
		
		monoSpecTerm = d*g*f/(4.0h*nv*nl);//a2 / (max(0.1f, lh*lh) * (roughness + 0.5f) * (d * d) * 4);
		if(nv<=0.0f || nl<=0.0f){monoSpecTerm = 0.0f;} 
		float3 compF0 = mitsuba_Fresnel(1.0f,eta)*specColor;
		if(_use_SMS>0.0){
			float mse = computeMSE(nv, a);
			monoSpecTerm = monoSpecTerm+monoSpecTerm*mse*compF0;//*nl;//maybe add cosine of theta
		}
		specularTerm = specColor*monoSpecTerm;
	}
	else{
		if(_GGXVG>0.0f){
		specularTerm = specularTerm + MSVG_evalWithPdf(local_light_dir, local_view_dir, floatspeccolor, a, eta, 1, 1);}
		else{
		specularTerm = specularTerm + MSVG_evalWithPdf(local_light_dir, local_view_dir, floatspeccolor, a, eta, 5, 1);
		}
		//specularTerm = specularTerm/Frame_cosTheta(local_view_dir);
	}
	
	

    // on mobiles (where half actually means something) denominator have risk of overflow
    // clamp below was added specifically to "fix" that, but dx compiler (we convert bytecode to metal/gles)
    // sees that specularTerm have only non-negative terms, so it skips max(0,..) in clamp (leaving only min(100,...))
#if defined (SHADER_API_MOBILE)
    specularTerm = float3(1,1,1)*specularTerm - float3(1,1,1)*1e-4f;
#endif

#ifdef UNITY_COLORSPACE_GAMMA
    //specularTerm = sqrt(max(1e-4f, specularTerm));
#endif


#if defined (SHADER_API_MOBILE)
    specularTerm = float3(1,1,1)*clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
#endif
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = float3(0,0,0);
#endif
	if(_plot_mode>0.0f){return float4(pow(((1.0/3.14159)*diffColor + /*(1.0/3.14159)*/ specularTerm),1),1);}
	half3 color =   ((1.0/3.14159)*diffColor + /*(1.0/3.14159)*/ specularTerm) * light.color * nl;
                    //+ gi.diffuse * diffColor
                    //+ surfaceReduction * gi.specular * specColor * f;
	//if(_show_info_shader){color = color + INFORMATIVE_SHADER(diffColor, specColor, oneMinusReflectivity, smoothness, normal, viewDir, light, gi, F0, tex_xy);}
    return half4(color,1);
}







/*This is a port from the mysterious merl.h I have in my old Mitsuba at school*/
float3 getAnglesFromVect(float3 vv, float3 hv){
	float cosTH = Frame_cosTheta(hv);
	float sinTH = Frame_sinTheta(hv);
	float cosPH = Frame_cosPhi(hv);
	float sinPH = Frame_sinPhi(hv);
	
	float tmpX = vv[0] * cosPH + vv[1] * sinPH;
	float3 transformedWi = Unity_SafeNormalize(float3(tmpX * cosTH - vv[2] * sinTH,vv[1] * cosPH - vv[0] * sinPH,vv[2] * cosTH + tmpX * sinTH));
	
	float phiD = acos(Frame_cosPhi(transformedWi));
	float thetaD = acos(Frame_cosTheta(transformedWi));
	float thetaH = acos(Frame_cosTheta(hv));
	return float3(phiD,thetaD,thetaH);
	
}



uniform float4 _MainTex_TexelSize;

float _FloatMERL;

sampler2D_float _MERLTex;
sampler2D_float _MERLLogTex;
sampler2D_float _MERLTexF;
sampler2D_float _MERLLogTexF;



float3 get_merl_simple(float theta_h_index, float theta_d_index, float phi_d_index){
	//Pull the correct MERL value, no interpolation
	float2 uv = float2((theta_h_index*90.0f+theta_d_index+0.5f)/8192.0f,1.0f-(phi_d_index+0.5f)/256.0f);
	return tex2D(_MERLTex, uv) * pow(2.0f,tex2D(_MERLLogTex, uv)*16.0f);
}

float3 get_merl_float(float theta_h_index, float theta_d_index, float phi_d_index){
	//Pull the correct MERL value, no interpolation
	float2 uv = float2((theta_h_index*90.0f+theta_d_index+0.5f)/8192.0f,1.0f-(phi_d_index+0.5f)/256.0f);
	float3 ex = tex2D(_MERLLogTexF, uv);
	return (1.0f+tex2D(_MERLTexF, uv)) * pow(2.0f,ex*256.0-240.0);
}

float3 get_merl_smart(float theta_h_index, float theta_d_index, float phi_d_index){
	if(_FloatMERL>0.0f){return get_merl_float(theta_h_index, theta_d_index, phi_d_index);}
	else{return get_merl_simple(theta_h_index, theta_d_index, phi_d_index);}
}

float _InterpMERL;

half4 BRDF6_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0)){
		
		if(_plot_mode>0.0f){
		//return half4(tex_xy[0],0,tex_xy[1],1);
		normal = float3(0,0,1);
		float halfTheta = (1-tex_xy[0])*(UNITY_PI/2.0f);
		float halfDiff = (1-tex_xy[1])*(UNITY_PI/2.0f);
		//float3 halfDir = float3(sin(halfTheta),0,cos(halfTheta));
		light.dir = float3(sin(halfDiff),0,cos(halfDiff));
		viewDir = float3(-sin(halfDiff),0,cos(halfDiff));
		normal = float3(0,sin(halfTheta),cos(halfTheta));
		light.color = float3(1,1,1);
	}
		
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 uvec;
	float3 vvec;
	float3 local_normal = float3(0,0,1);
	if(nz<0.0f)
	{
		float a = 1.0f / (1.0f - nz);
		float b = nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, -b, nx);
		vvec = float3(b, ny * ny*a - 1.0f, -ny);
	}
	else{
		float a = 1.0f / (1.0f + nz);
		float b = -nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, b, -nx);
		vvec = float3(b, 1.0f - ny * ny * a, -ny);
	}
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	float nl = dot(float3(0,0,1), local_light_dir);
	float nv = dot(float3(0,0,1), local_view_dir);
	
	float twi = acos(local_light_dir[2]);
    float two = acos(local_view_dir[2]);
    float pwi = atan2(local_light_dir[1], local_light_dir[0]);
    float pwo = atan2(local_view_dir[1],local_view_dir[0]);

	
	float4 thetaphiangles = std_coords_to_half_diff_coords(two,pwo,twi,pwi);
	float3 coordinates = float3(thetaphiangles[3],thetaphiangles[2],thetaphiangles[0]);
	//coordinates = getAnglesFromVect(local_light_dir,local_half_dir);
    //return float4(float3(getThetaHIndex(coordinates[2])/89.0f,getThetaDIndex(coordinates[1])/89.0f,getPhiDIndex(coordinates[0])/179.0f),1);
	
	float thetaHraw = getThetaHIndex_raw(coordinates[2]);
	float thetaDraw = getThetaDIndex_raw(coordinates[1]);
	float phiDraw = getPhiDIndex_raw(coordinates[0]);
	
	float thetaHfloor = floor(thetaHraw);
	float thetaDfloor = floor(thetaDraw);
	float phiDfloor = floor(phiDraw);
	float3 merl_value;
	if(_InterpMERL == 0.0f){
		merl_value = get_merl_smart(getThetaHIndex(coordinates[2]),getThetaDIndex(coordinates[1]),getPhiDIndex(coordinates[0]));
	}
	else{
		//compute the spillover
		float thetaHextra=thetaHraw-thetaHfloor;
		float thetaDextra=thetaDraw-thetaDfloor;
		float phiDextra=phiDraw-phiDfloor;
		//Don't spill into the next block
		if(thetaHfloor>88.0f){thetaHextra=0.0f;}
		if(thetaDfloor>88.0f){thetaDextra=0.0f;}
		if(phiDfloor>178.0f){phiDextra=0.0f;}
		//Trilinear Interpolation!
		float3 lll = get_merl_smart(thetaHfloor,thetaDfloor,phiDfloor);
		float3 llr = get_merl_smart(thetaHfloor,thetaDfloor,phiDfloor+1);
		float3 lrl = get_merl_smart(thetaHfloor,thetaDfloor+1,phiDfloor);
		float3 lrr = get_merl_smart(thetaHfloor,thetaDfloor+1,phiDfloor+1);
		float3 rll = get_merl_smart(thetaHfloor+1,thetaDfloor,phiDfloor);
		float3 rlr = get_merl_smart(thetaHfloor+1,thetaDfloor,phiDfloor+1);
		float3 rrl = get_merl_smart(thetaHfloor+1,thetaDfloor+1,phiDfloor);
		float3 rrr = get_merl_smart(thetaHfloor+1,thetaDfloor+1,phiDfloor+1);
		// i means interpolated
		float3 ill = lll*(1-thetaHextra) + rll*(thetaHextra);
		float3 ilr = llr*(1-thetaHextra) + rlr*(thetaHextra);
		float3 irl = lrl*(1-thetaHextra) + rrl*(thetaHextra);
		float3 irr = lrr*(1-thetaHextra) + rrr*(thetaHextra);
		
		float3 iil = ill*(1-thetaDextra) + irl*(thetaDextra);
		float3 iir = ilr*(1-thetaDextra) + irr*(thetaDextra);
		
		merl_value = iil*(1-phiDextra) + iir*(phiDextra);
		
	}
	
	if(_plot_mode>0.0f){return float4(pow(merl_value,1),1);}
	merl_value = merl_value*light.color*nl;
	//if(merl_value[2]<0.01f){merl_value[0]=1.0f;}
	
	//if(_show_info_shader){merl_value = merl_value + INFORMATIVE_SHADER(diffColor, specColor, oneMinusReflectivity, smoothness, normal, viewDir, light, gi, F0, tex_xy);}
		
		return float4(merl_value,1);
	}
	
	
	half4 BRDF8_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0)){
		
	if(_plot_mode>0.0f){
		//return half4(tex_xy[0],0,tex_xy[1],1);
		normal = float3(0,0,1);
		float halfTheta = (1-tex_xy[0])*UNITY_PI/2.0f;
		float halfDiff = (1-tex_xy[1])*UNITY_PI/2.0f;
		//float3 halfDir = float3(sin(halfTheta),0,cos(halfTheta));
		light.dir = normalize(float3(sin(halfDiff),0,cos(halfDiff)));
		viewDir = normalize(float3(-sin(halfDiff),0,cos(halfDiff)));
		normal = normalize(float3(0,sin(halfTheta),cos(halfTheta)));
		//light.color = float3(0.5,0.5,0.5);
	}
		
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 uvec;
	float3 vvec;
	float3 local_normal = float3(0,0,1);
	if(nz<0.0f)
	{
		float a = 1.0f / (1.0f - nz);
		float b = nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, -b, nx);
		vvec = float3(b, ny * ny*a - 1.0f, -ny);
	}
	else{
		float a = 1.0f / (1.0f + nz);
		float b = -nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, b, -nx);
		vvec = float3(b, 1.0f - ny * ny * a, -ny);
	}
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	float nl = dot(float3(0,0,1), local_light_dir);
	if(nl<0.0f) return float4(0,0,0,1);
	float nv = dot(float3(0,0,1), local_view_dir);
	
	float twi = acos(local_light_dir[2]);
    float two = acos(local_view_dir[2]);
    float pwi = atan2(local_light_dir[1], local_light_dir[0]);
    float pwo = atan2(local_view_dir[1],local_view_dir[0]);

	
	float4 thetaphiangles = std_coords_to_half_diff_coords(two,pwo,twi,pwi);
	float3 coordinates = float3(thetaphiangles[3],thetaphiangles[2],thetaphiangles[0]);
	//coordinates = getAnglesFromVect(local_light_dir,local_half_dir);
    //return float4(float3(getThetaHIndex(coordinates[2])/89.0f,getThetaDIndex(coordinates[1])/89.0f,getPhiDIndex(coordinates[0])/179.0f),1);
	
	float thetaHraw = getThetaHIndex_raw(coordinates[2]);
	float thetaDraw = getThetaDIndex_raw(coordinates[1]);
	float phiDraw = getPhiDIndex_raw(coordinates[0]);
	float3 color = float3(coordinates[2]/(2.0f/UNITY_PI),0,coordinates[1]/(2.0f/UNITY_PI));
	//if(_show_info_shader){color = color + INFORMATIVE_SHADER(diffColor, specColor, oneMinusReflectivity, smoothness, normal, viewDir, light, gi, F0, tex_xy);}
	return float4(color * light.color,1);//phiDraw/180.0f,1);
	}
	
	
	
	half4 BRDF9_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0)){
		
	if(_plot_mode>0.0f){
		//return half4(tex_xy[0],0,tex_xy[1],1);
		normal = float3(0,0,1);
		float halfTheta = (1-tex_xy[0])*UNITY_PI/2.0f;
		float halfDiff = (1-tex_xy[1])*UNITY_PI/2.0f;
		//float3 halfDir = float3(sin(halfTheta),0,cos(halfTheta));
		light.dir = normalize(float3(sin(halfDiff),0,cos(halfDiff)));
		viewDir = normalize(float3(-sin(halfDiff),0,cos(halfDiff)));
		normal = normalize(float3(0,sin(halfTheta),cos(halfTheta)));
		//light.color = float3(0.5,0.5,0.5);
	}
		
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 uvec;
	float3 vvec;
	float3 local_normal = float3(0,0,1);
	if(nz<0.0f)
	{
		float a = 1.0f / (1.0f - nz);
		float b = nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, -b, nx);
		vvec = float3(b, ny * ny*a - 1.0f, -ny);
	}
	else{
		float a = 1.0f / (1.0f + nz);
		float b = -nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, b, -nx);
		vvec = float3(b, 1.0f - ny * ny * a, -ny);
	}
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	float nl = dot(float3(0,0,1), local_light_dir);
	if(nl<0.0f) return float4(0,0,0,1);
	float nv = dot(float3(0,0,1), local_view_dir);
	
	float twi = acos(local_light_dir[2]);
    float two = acos(local_view_dir[2]);
    float pwi = atan2(local_light_dir[1], local_light_dir[0]);
    float pwo = atan2(local_view_dir[1],local_view_dir[0]);

	
	float4 thetaphiangles = std_coords_to_half_diff_coords(two,pwo,twi,pwi);
	float3 coordinates = float3(thetaphiangles[3],thetaphiangles[2],thetaphiangles[0]);
	//coordinates = getAnglesFromVect(local_light_dir,local_half_dir);
    //return float4(float3(getThetaHIndex(coordinates[2])/89.0f,getThetaDIndex(coordinates[1])/89.0f,getPhiDIndex(coordinates[0])/179.0f),1);
	
	float thetaHraw = coordinates[2];
	float thetaDraw = coordinates[1];
	float phiDraw = coordinates[0];
	float3 infocolor = float3(0.3,0.3,0.3);
	if(thetaHraw*(2.0f/UNITY_PI)<0.1){infocolor[0]=1.0f;}
	if(thetaDraw*(2.0f/UNITY_PI)>0.8){infocolor[1]=1.0f;}
	if(thetaHraw*(2.0f/UNITY_PI)>0.8){infocolor[2]=1.0f;}
	
	
	return float4(infocolor*light.color,1);//phiDraw/180.0f,1);
	}
	
	
	
	half4 BRDF7_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0))
{
	//return float4(displayscale(normal,2),1);
	//normal = norm(normal);
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);
	//return float4(displayscale(halfDir,2),1);
    half nl = saturate(dot(normal, light.dir));
    float nh = saturate(dot(normal, halfDir));
    half nv = saturate(dot(normal, viewDir));
    float lh = saturate(dot(light.dir, halfDir));
	float vh = saturate(dot(viewDir,halfDir));
	
	/*Get local shading frame directions for view, light, and half vectors
	Algorithm taken from Duff et al. "Building an Orthonormal Basis, Revisited",
	Journal of Computer Graphics, Vol. 6, No. 1, 2017*/
	
	
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 uvec;
	float3 vvec;
	float3 local_normal = float3(0,0,1);
	if(nz<0.0f)
	{
		float a = 1.0f / (1.0f - nz);
		float b = nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, -b, nx);
		vvec = float3(b, ny * ny*a - 1.0f, -ny);
	}
	else{
		float a = 1.0f / (1.0f + nz);
		float b = -nx * ny * a;
		uvec = float3(1.0f - nx * nx * a, b, -nx);
		vvec = float3(b, 1.0f - ny * ny * a, -ny);
	}
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	
//	float3 transformed_normal = Unity_SafeNormalize(mul(Tmatrix,normal));
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	

	
	nl = dot(float3(0,0,1), local_light_dir);
	nv = dot(float3(0,0,1), local_view_dir);
	
    //half perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    //half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
	
	//Core BRDF code starts here
	float a = smoothness;
	if(_albedo_from_floats){
	diffColor = half3(_dr,_dg,_db);
	specColor = half3(_sr,_sg,_sb);
	a = _RoughnessAlpha;
	}
	float3 floatspeccolor = float3(_sr,_sg,_sb);
	//return float4(floatspeccolor,1);
	float eta = _FresnelEta;

	float3 specularTerm = MSVG_evalWithPdf(local_light_dir, local_view_dir, floatspeccolor, a, eta, 5, 1);
	
	

    // on mobiles (where half actually means something) denominator have risk of overflow
    // clamp below was added specifically to "fix" that, but dx compiler (we convert bytecode to metal/gles)
    // sees that specularTerm have only non-negative terms, so it skips max(0,..) in clamp (leaving only min(100,...))
	
	half3 color =   ((1.0/3.14159)*diffColor + /*(1.0/3.14159)*/ specularTerm * specColor) * light.color * nl;
                    //+ gi.diffuse * diffColor
                    //+ surfaceReduction * gi.specular * specColor * f;
    return half4(color,1);
}

//float4x4 unity_WorldToObject;
float _rotx,_roty,_rotz;
float _ax,_ay,_az;
float4x4 _rotmat;

half4 BRDF_ACT_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0))
{ 
	
	
	float3 localnormal = Unity_SafeNormalize(mul(_rotmat,float4(normal,1)).xyz);
	float3 locallight = Unity_SafeNormalize(mul(_rotmat,float4(light.dir,1)).xyz);
	float3 localview = Unity_SafeNormalize(mul(_rotmat,float4(viewDir,1)).xyz);
	
	//return float4(localnormal,1);
	
	normal = localnormal;
	light.dir = locallight;
	viewDir = localview;
	
	float3 light_xy = float3(locallight[0],locallight[1],0);//this is going to get normalized which could be bad!
	float3 view_xy = float3(localview[0],localview[1],0);
	
	light_xy = Unity_SafeNormalize(light_xy);
	view_xy = Unity_SafeNormalize(view_xy);
	
	//return float4(localnormal[0],localnormal[1],localnormal[2],1);
	//return float4(displayscale(normal,2),1);
	//normal = norm(normal);
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);
	//return float4(displayscale(halfDir,2),1);
    half nl = saturate(dot(normal, light.dir));
    float nh = saturate(dot(normal, halfDir));
    half nv = saturate(dot(normal, viewDir));
    float lh = saturate(dot(light.dir, halfDir));
	float vh = saturate(dot(viewDir,halfDir));
	
	/*Get local shading frame directions for view, light, and half vectors
	Algorithm taken from Duff et al. "Building an Orthonormal Basis, Revisited",
	Journal of Computer Graphics, Vol. 6, No. 1, 2017*/
	
	
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 tvec = float3(_ax,_ay,_az);
	
	float3 uvec = cross(normal,tvec);
	float3 vvec = cross(normal,uvec);
	float3 local_normal = float3(0,0,1);
	//if(nz<0.0f)
	//For anisotropy
	
	
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	
//	float3 transformed_normal = Unity_SafeNormalize(mul(Tmatrix,normal));
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_light_xy = Unity_SafeNormalize(float3(local_light_dir.x,local_light_dir.y,0));
	float old_angle_to_z = acos(light_xy.x);
	float new_angle_to_z = acos(local_light_xy.x);
	float angle_diff = old_angle_to_z - new_angle_to_z;
	float na = -angle_diff;
	//if(normal.z >0) na = 0;
	float3x3 zrot = float3x3(cos(na),-sin(na),0,sin(na),-cos(na),0,0,0,1);
	//float3x3 zrot = float3x3(cos(na),sin(na),0,-sin(na),-cos(na),0,0,0,1);
	//local_light_dir = mul(zrot,local_light_dir);
	
	
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	//local_view_dir = mul(zrot,local_view_dir);
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	

	
	nl = dot(float3(0,0,1), local_light_dir);
	nv = dot(float3(0,0,1), local_view_dir);
	
    //half perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    //half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
	
diffColor = half3(_dr,_dg,_db);
	specColor = half3(_sr,_sg,_sb);
	float eta = _FresnelEta;
    float ax = _RoughnessAlpha;
	float ay = _RoughnessAlpha*_Anisotropic;
	float d;
	d = AniCTTermMitsuba(local_half_dir, ax, ay);
	//float g_smaller = min(nv,nl);
	//float g = min(1.0f,2.0*nh*g_smaller/vh);
	float g = mitsuba_G_Beckmann(local_light_dir,local_view_dir,local_half_dir,ax);
	float f = mitsuba_Fresnel(vh,eta);
    float specularTerm = d*g*f;//a2 / (max(0.1f, lh*lh) * (roughness + 0.5f) * (d * d) * 4);


    // on mobiles (where half actually means something) denominator have risk of overflow
    // clamp below was added specifically to "fix" that, but dx compiler (we convert bytecode to metal/gles)
    // sees that specularTerm have only non-negative terms, so it skips max(0,..) in clamp (leaving only min(100,...))
#if defined (SHADER_API_MOBILE)
    specularTerm = specularTerm - 1e-4f;
#endif



#if defined (SHADER_API_MOBILE)
    specularTerm = clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
#endif
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = 0.0;
#endif


    half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));
    half3 color =   ((1.0f/3.14159f)*diffColor + specularTerm * specColor) * light.color * nl;
                    //+ gi.diffuse * diffColor
                    //+ surfaceReduction * gi.specular * specColor * f;
	if(_plot_mode>0.0f){return float4(pow(((1.0/3.14159)*diffColor + /*(1.0/3.14159)*/ specularTerm* specColor),1),1);}
	//if(_show_info_shader){color = color + INFORMATIVE_SHADER(diffColor, specColor, oneMinusReflectivity, smoothness, normal, viewDir, light, gi, F0, tex_xy);}
    return half4(pow(color, 1.0), 1);
}



half4 BRDF_AGGX_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0))
{
	
	float3 localnormal = Unity_SafeNormalize(mul(_rotmat,float4(normal,1)).xyz);
	float3 locallight = Unity_SafeNormalize(mul(_rotmat,float4(light.dir,1)).xyz);
	float3 localview = Unity_SafeNormalize(mul(_rotmat,float4(viewDir,1)).xyz);
	
	//return float4(localnormal,1);
	
	normal = localnormal;
	light.dir = locallight;
	viewDir = localview;
	
	float3 light_xy = float3(locallight[0],locallight[1],0);//this is going to get normalized which could be bad!
	float3 view_xy = float3(localview[0],localview[1],0);
	
	light_xy = Unity_SafeNormalize(light_xy);
	view_xy = Unity_SafeNormalize(view_xy);
	
	//return float4(localnormal[0],localnormal[1],localnormal[2],1);
	//return float4(displayscale(normal,2),1);
	//normal = norm(normal);
    float3 halfDir = Unity_SafeNormalize (float3(light.dir) + viewDir);
	//return float4(displayscale(halfDir,2),1);
    half nl = saturate(dot(normal, light.dir));
    float nh = saturate(dot(normal, halfDir));
    half nv = saturate(dot(normal, viewDir));
    float lh = saturate(dot(light.dir, halfDir));
	float vh = saturate(dot(viewDir,halfDir));
	
	/*Get local shading frame directions for view, light, and half vectors
	Algorithm taken from Duff et al. "Building an Orthonormal Basis, Revisited",
	Journal of Computer Graphics, Vol. 6, No. 1, 2017*/
	
	
	float nx = normal[0];
	float ny = normal[1];
	float nz = normal[2];
	
	float3 xaxis = float3(1,0,0);
	float3 yaxis = float3(0,1,0);
	float3 zaxis = float3(0,0,1);
	
	float3 tvec = float3(_ax,_ay,_az);
	
	float3 uvec = cross(normal,tvec);
	float3 vvec = cross(normal,uvec);
	float3 local_normal = float3(0,0,1);
	//if(nz<0.0f)
	//For anisotropy
	
	
	
	uvec=Unity_SafeNormalize(uvec);
	vvec=Unity_SafeNormalize(vvec);
	
	float3x3 Tmatrix = float3x3(uvec[0],uvec[1],uvec[2],vvec[0],vvec[1],vvec[2],normal[0],normal[1],normal[2]);
	
	
//	float3 transformed_normal = Unity_SafeNormalize(mul(Tmatrix,normal));
	
	float3 local_light_dir = Unity_SafeNormalize(mul(Tmatrix,light.dir));
	float3 local_light_xy = Unity_SafeNormalize(float3(local_light_dir.x,local_light_dir.y,0));
	float old_angle_to_z = acos(light_xy.x);
	float new_angle_to_z = acos(local_light_xy.x);
	float angle_diff = old_angle_to_z - new_angle_to_z;
	float na = -angle_diff;
	//if(normal.z >0) na = 0;
	float3x3 zrot = float3x3(cos(na),-sin(na),0,sin(na),-cos(na),0,0,0,1);
	//float3x3 zrot = float3x3(cos(na),sin(na),0,-sin(na),-cos(na),0,0,0,1);
	//local_light_dir = mul(zrot,local_light_dir);
	
	
	float3 local_view_dir = Unity_SafeNormalize(mul(Tmatrix,viewDir));
	//local_view_dir = mul(zrot,local_view_dir);
	
	float3 local_half_dir = Unity_SafeNormalize(local_light_dir + local_view_dir);
	

	
	nl = dot(float3(0,0,1), local_light_dir);
	nv = dot(float3(0,0,1), local_view_dir);
	
    //half perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
    //half roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
	
	//Core BRDF code starts here
	float a = smoothness;
	if(_albedo_from_floats){
	diffColor = half3(_dr,_dg,_db);
	specColor = half3(_sr,_sg,_sb);
	a = _RoughnessAlpha;
	}
	float3 floatspeccolor = float3(_sr,_sg,_sb);
	//return float4(floatspeccolor,1);
	float eta = _FresnelEta;
	float d;
	float g;
	float f;
	float monoSpecTerm = 0.0f;
	float3 specularTerm = float3(0,0,0);
	
	
	
		
		d = GGXTermMitsuba(local_half_dir,a,_Anisotropic);
		
		if(_GGXVG>0.0f){g = GGXVG_G(local_light_dir,local_view_dir,local_half_dir);}
		else{g = mitsuba_G(local_light_dir,local_view_dir,local_half_dir,a);}

		f = mitsuba_Fresnel(vh,eta);
		
		monoSpecTerm = d*g*f/(4.0h*nv*nl);//a2 / (max(0.1f, lh*lh) * (roughness + 0.5f) * (d * d) * 4);
		if(nv<=0.0f || nl<=0.0f){monoSpecTerm = 0.0f;} 
		float3 compF0 = mitsuba_Fresnel(1.0f,eta)*specColor;
		specularTerm = specColor*monoSpecTerm;
	
	
	
	

    // on mobiles (where half actually means something) denominator have risk of overflow
    // clamp below was added specifically to "fix" that, but dx compiler (we convert bytecode to metal/gles)
    // sees that specularTerm have only non-negative terms, so it skips max(0,..) in clamp (leaving only min(100,...))
#if defined (SHADER_API_MOBILE)
    specularTerm = float3(1,1,1)*specularTerm - float3(1,1,1)*1e-4f;
#endif

#ifdef UNITY_COLORSPACE_GAMMA
    //specularTerm = sqrt(max(1e-4f, specularTerm));
#endif


#if defined (SHADER_API_MOBILE)
    specularTerm = float3(1,1,1)*clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
#endif
#if defined(_SPECULARHIGHLIGHTS_OFF)
    specularTerm = float3(0,0,0);
#endif
	if(_plot_mode>0.0f){return float4(pow(((1.0/3.14159)*diffColor + /*(1.0/3.14159)*/ specularTerm),1),1);}
	half3 color =   ((1.0/3.14159)*diffColor + /*(1.0/3.14159)*/ specularTerm) * light.color * nl;
                    //+ gi.diffuse * diffColor
                    //+ surfaceReduction * gi.specular * specColor * f;
	//if(_show_info_shader){color = color + INFORMATIVE_SHADER(diffColor, specColor, oneMinusReflectivity, smoothness, normal, viewDir, light, gi, F0, tex_xy);}
    return half4(color,1);
}



half4 DEBUGBRDF_Unity_PBS (half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
    float3 normal, float3 viewDir,
    UnityLight light, UnityIndirect gi, half F0, float2 tex_xy = float2(0,0))
{ return float4(1,1,1,1);}



// Include deprecated function
#define INCLUDE_UNITY_STANDARD_BRDF_DEPRECATED
#include "UnityDeprecated.cginc"
#undef INCLUDE_UNITY_STANDARD_BRDF_DEPRECATED

#endif // UNITY_STANDARD_BRDF_INCLUDED
