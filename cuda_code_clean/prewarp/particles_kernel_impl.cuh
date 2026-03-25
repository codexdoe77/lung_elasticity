/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
#include "read_dicom_data.h"
#include "constants.h"

// simulation parameters in constant memory
__constant__ SimParams params;



// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.voxelSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.voxelSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.voxelSize.z);
    return gridPos;
}


// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction,
                      float att_mul,
                      float spring)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + 1.1f*radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force += spring * (dist - collideDist) * norm;
        // dashpot (damping) force
        force += params.damping * relVel;
        // tangential shear force
        force += params.shear * tanVel;
        // attraction
        force += att_mul*attraction*relPos;
    }

    return (force / 10.f);
}


__global__
void collideD(float4 *newPos,               // output: new positions
              float4 *newVel,               // output: new velocities
              float4 *newForce,             // output: new forces
              int    *segment,
              uint    dynParticles,         // input: number of dynamic particles
              float   deltaTime)            // input: timestep
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= dynParticles) return;

    // read particle data from sorted arrays
    float4 fetchpos = FETCH(newPos, index);
    float3 pos = make_float3(fetchpos.x, fetchpos.y, fetchpos.z);
    float3 vel = make_float3(FETCH(newVel, index));

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    // collide with cursor sphere
    float3 colliderPos = params.colliderPos;
    float colliderIncrement = 0.f;

    int c = 0;
    do
    {
        float torLength = params.colliderInnerRadius[c] + params.colliderInnerRadius[c+1];
        float torFraction = colliderIncrement / torLength;
        if (params.colliderLength == 1)
            torFraction = 0.f;

        float3 relPos = pos - colliderPos;
        float dist = length(relPos);

        float3 projection = dot(relPos, params.colliderOrientation) * params.colliderOrientation;
        float projLen = length(projection);

        float3 rejection = relPos - projection;
        float rejLen = length(rejection);

        float torRadius = (1.f - torFraction)*params.colliderOuterRadius[c] + torFraction*params.colliderOuterRadius[c+1];
        float torHeight = (1.f - torFraction)*params.colliderInnerRadius[c] + torFraction*params.colliderInnerRadius[c+1];

        if ( rejLen > (torRadius - torHeight - params.particleRadius) &&
                rejLen < (torRadius + torHeight + params.particleRadius) &&
                projLen < (torHeight + params.particleRadius))
        {
            float3 rejNorm = rejection / rejLen;
            float3 sphere = colliderPos + torRadius * rejNorm;

            force += collideSpheres(pos, sphere, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius,
                                                    torHeight, 0.0f, 0.0f, 1.0f);

            float3 rejPos = sphere - pos;
            float rejDist = length(rejPos);
            if (rejDist < (params.particleRadius + torHeight))
            {
                float3 displace = pos - sphere;
                float3 dispNorm = displace / length(displace);
                displace = sphere + (params.particleRadius + torHeight)*dispNorm - pos;
                pos += displace;
            }
        }
        //colliderPos += 0.5f * params.colliderOrientation * (params.colliderHeight[c/2] + params.colliderHeight[(c/2)+1]);
        colliderIncrement += length(params.colliderOrientation * params.particleRadius);
        if (colliderIncrement >= torLength)
        {
            c++;
            colliderIncrement = 0.f;
        }
        colliderPos += params.colliderOrientation * params.particleRadius;
    }
    while (params.colliderLength > 1 && c < (params.colliderLength - 1));

    newForce[index] = make_float4( force, 0.0f);
    newPos[index] = make_float4( pos, fetchpos.w);
}


// create a color ramp
__device__
float3 colorRamp(float t)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 0.1, 0.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 1.0, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 0.0, 0.0, }
    };
    t = t * __int2float_rn(ncolors-1);
    int i = __float2int_rd(t);
    float u = t - __int2float_rn(i);
    float3 rgb;
    rgb.x = lerp(c[i][0], c[i+1][0], u);
    rgb.y = lerp(c[i][1], c[i+1][1], u);
    rgb.z = lerp(c[i][2], c[i+1][2], u);

    return(rgb);
}


// collide two spheres using DEM method
__device__
float3 connectParticles(float3 len, float *disp,
                        float3 posA,
                        float3 posB,
                        float3 velA, float3 velB,
                        float springConstant, float springDamping,
                        float elasticModulus, float shearModulus,
                        float energyModulus,
                        float potential, float2 stretchiness,
                        uint switcher )
{
    // calculate relative position
    float3 relPos = posB - posA;
    float3 displacement = len - relPos;

    // distance between particles
    float dist = length(relPos);

    // rest length distance
    float rest = length(len);

    // variable to sum force
    float3 force = make_float3(0.0f);

  // Calculate dot product between displacement and rest vector, find projection and rejection of displacement onto rest vector
    float3 projection = len * dot(relPos, len) / rest / rest;
    float3 rejection = relPos - projection;

  //Linearly Elastic Spring Force (Hooke's Law)
    // force += (springConstant * (dist - rest) / rest) * (relPos);

  //Elastic Modulus Force
    force += (elasticModulus * (length(projection) - rest) / rest) * (projection);

  //Shear Modulus Force
    //shearModulus = shearModulus / 1000000.f; //Updates units to kPa (theoretically)
    force += 1.f * (shearModulus * length(rejection) / rest) * rejection; //COMMENTED OUT FOR TEST CASE?

  //Lennard-Jones bi-reciprocal function
    //  Documented in 'Physically Based Deformable Models in Computer Graphics' by Andrew Nealen et al.
    //  See DeformationNealen.pdf in doc folder.
    // if (stretchiness.x > 0 && stretchiness.y > 0)
        // force += -1.f * ( potential * (stretchiness.x*powf(rest,stretchiness.y) - stretchiness.y*powf(rest,stretchiness.x)) / (dist * abs(stretchiness.x - stretchiness.y)) ) * displacement;

  // Strain Energy
    float energy = 0.5f * energyModulus * dot(displacement,displacement) / dot(len,len);
    if (length(displacement) > 0.01f * params.voxelSize.x)
    {
        // force += -energy * (1.f - springDamping) * displacement / length(displacement);
    }

  // Spring damping
    float3 relVel = velB - velA;
    force += springDamping * relVel; //COMMENTED OUT FOR TEST CASE

    // record either force magnitude or spring displacement for visualization colorRamp
    if (switcher == 1)
    {
        *disp += dist - rest;
    }
    else
    {
        *disp += length(force);
    }

    return force;
}


__global__
void staticD( float4  *oldPos,               // input: positions
              float4  *oldVel,               // input: velocities
              float4  *newForce,             // output: new forces
              float4  *staticPos,
              float   *elasticity,
              uint    *cons,                 // input: particle connection indices
              uint    *conCount,             // input: number of connections per particle
              uint    *conStart,             // input: where connection indices start for each particle
              float4  *restLengths,          // input: particle connection lengths
              uint     dynParticles,
              uint     statParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= dynParticles) return;

    // read particle data from sorted arrays
    float elasticModulus = elasticity[index];
    float3 posA = make_float3(FETCH(oldPos, index));
    float3 velA = make_float3(FETCH(oldVel, index));
    float3 force = make_float3(0.0f);

    float disp = 0.0f;
    uint springStart = conStart[index];
    uint springCount = conCount[index];
    for (uint i=0; i<springCount; i++)
    {
        int con = cons[springStart + i];

        float3 len;
        len.x = restLengths[springStart + i].x;
        len.y = restLengths[springStart + i].y;
        len.z = restLengths[springStart + i].z;

        float3 posB, velB;
        posB = make_float3(FETCH(staticPos, con));
        velB = make_float3(0.f,0.f,0.f);

        force += connectParticles(len, &disp, posA, posB, velA, velB, params.springStrength, params.springDamping,
                                        elasticModulus, params.shearModulus, params.energyModulus,
                                        params.minPotential, params.stretchiness, params.colorSwitch);
    }

    // write new force back to original unsorted location
    newForce[index] = 0.5f*make_float4(force, 0.0f);
}


//Used as kernel in connections class (particleSystem_cuda.cu)
__global__
void connectD(float4  *tempPos,
              float4  *oldPos,               // input: positions
              float4  *oldVel,               // input: velocities
              float4  *sForce,               // output: new forces
              float  *sStretch,              // output: spring stretch
              float4  *tForce,               // output: new forces
              float   *elasticity,           // input: elasticity array
              uint    *cons,                 // input: particle connection indices
              uint    *conCount,
              uint    *conStart,
              float4  *restLengths,          // input: particle connection lengths
              int     *segment,              // input: scaled intensity values
              float   *cntrScale,
              uint     dynParticles,
              float4  *newColor,
              uint3   dataSize,
              float4  *initPos,
              float4  *finalPos,
              int   *staticTrue,
              bool    m_bAnchor,
              int   *surface)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= dynParticles) return;


    uint s = (int) floorf( powf((float) dynParticles, 1.0f / 3.0f) + 0.5f);

    // if (m_bAnchor)
    {
        // if (surface[index] == 1)
        if (staticTrue[index] == 1.f)
        {
            sForce[index] = make_float4(0.f, 0.f, 0.f, 0.f);
            initPos[index] = make_float4(0.f, 0.f, 0.f, 0.f);
            sStretch[index] = 0.f;
            newColor[index] = make_float4(0.f, 0.f, 0.f, 0.f);
            return;
        }
    }

    int segData = segment[index];

    uint cntr;
    if (segData == 1000) cntr = 0;
    else cntr = segData;

    float scalar = cntrScale[cntr] + 1.f;

    // read particle data from sorted arrays
    float elasticModulusA = elasticity[index];
    float3 posI = make_float3(FETCH(tempPos, index));
    float3 posA = make_float3(FETCH(oldPos, index));
    float3 velA = make_float3(FETCH(oldVel, index));
    float3 force = make_float3(FETCH(sForce, index));
    float springD = sStretch[index];
    force += params.particleMass*params.gravity;

    float3 dist = make_float3( (params.osc_origin.x - posI.x),
                                (params.osc_origin.y - posI.y),
                                (params.osc_origin.z - posI.z) );
    float dist_len = length(dist);

    float disp = 0.0f;
    uint springStart = conStart[index];
    uint springCount = conCount[index];
    for (uint i=0; i<springCount; i++)
    {
        int con = cons[springStart + i];
        if (con != 0xffffffff)
        {

        finalPos[con] = make_float4(0.f, 0.f, 0.f, 0.f);
        float3 len;
        len.x = restLengths[springStart + i].x;
        len.y = restLengths[springStart + i].y;
        len.z = restLengths[springStart + i].z;

        float3 posB, velB;
        posB = make_float3(FETCH(oldPos, con));
        posI = make_float3(FETCH(tempPos, con));
        float elasticModulusB = elasticity[con];
        //float elasticModulus = 0.5f*(elasticModulusA + elasticModulusB);

        float3 relPos = posB - posA;
        float distance = length(relPos);
        float rest = length(len);
        float springStretch = (distance - rest) / rest;

        float3 condist = make_float3( (params.osc_origin.x - posI.x),
                            (params.osc_origin.y - posI.y),
                            (params.osc_origin.z - posI.z) );
        float condist_len = length(dist);

        velB = make_float3(FETCH(oldVel, con));
        if (segment[con] == segData)
        {
            len.x *= scalar;
            len.y *= scalar;
            len.z *= scalar;
        }

        float3 conForce = connectParticles(len, &disp, posA, posB, velA, velB, params.springStrength, params.springDamping,
                                        elasticModulusB, params.shearModulus, params.energyModulus,
                                        params.minPotential, params.stretchiness, params.colorSwitch);

        force += conForce;
        springD += springStretch;
        springD /= springCount;
        finalPos[con] = make_float4(posI,0);
        }
    }

    // write new force back to original unsorted location
    sForce[index] += make_float4(force, 0.0f);
    sStretch[index] = springD;
    initPos[index] = make_float4(posA, 0.0f);

    // update particle color with respect to force or displacement
    float t;
    if (params.colorSwitch != 0)
    {
        if (params.colorSwitch == 1)
        {
            t = disp / (4.f * params.particleRadius);
            if (t < -1.f) t = -0.99f;
            if (t > 1.f)  t = 0.99f;
            t += 1.f;
            t /= 2.f;
        }
        else
        {

            t = (1.f) * elasticModulusA *0.5;
            if (t < 0.f) t = 0.f;
            if (t > 1.f)  t = 1.f;

        }
        float3 rgb = colorRamp(t);
        newColor[index] = make_float4( rgb, 0.5f );
    }
}



struct intConnect_functor
{
    float deltaTime;

    __host__ __device__
    intConnect_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        volatile float4 forSData = thrust::get<2>(t);
        volatile float4 forTData = thrust::get<3>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
        float3 force = make_float3(forSData.x, forSData.y, forSData.z); // + make_float3(forTData.x, forTData.y, forTData.z);

        // Explicit Euler
        vel += force * deltaTime;
        vel *= params.globalDamping;
        pos += vel * deltaTime;

        // // set this to zero to disable collisions with cube sides
        // float3 worldSize = 0.5f * params.worldSize;

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};



__global__
void shrinkD( float4  *oldPos,               // input: positions
              float4  *oldVel,               // input: velocities
              float4  *newForce,             // output: new forces
              uint    *cons,                 // input: particle connection indices
              float4  *restLengths,                 // input: particle connection lengths
              int    *segment,              // input: scaled intensity values
              float  *cntrScale,
              uint    dynParticles,
              uint    statParticles,
              float   range,
              float   deltaTime,
              float4  *newColor)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= dynParticles) return;

    int segData = segment[index];
    if (segData != params.contourSwitch) return;
    uint cntr = segData;

    float scalar = cntrScale[cntr] + 1.f;

    // read particle data from sorted arrays
    float3 velA = make_float3(FETCH(oldVel, index));
    float3 posA = make_float3(FETCH(oldPos, index));

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    float disp = 0.0f;
    for (uint i=0; i<NUM_SPRINGS; i++)
    {
        float indic = restLengths[NUM_SPRINGS*index + i].w;
        if (indic < 0.f) continue;

        int con = cons[NUM_SPRINGS*index + i];
        if (con == 0xffffffff) continue;
        if (segment[con] != segData) continue;

        float3 len;
        len.x = scalar * restLengths[NUM_SPRINGS*index + i].x;
        len.y = scalar * restLengths[NUM_SPRINGS*index + i].y;
        len.z = scalar * restLengths[NUM_SPRINGS*index + i].z;

        float3 velB = make_float3(FETCH(oldVel, con));
        float3 posB = make_float3(FETCH(oldPos, con));

        force += connectParticles(len, &disp, posA, posB, velA, velB, params.springStrength, params.springDamping,
                                        params.elasticModulus, params.shearModulus, params.energyModulus,
                                        params.minPotential, params.stretchiness, params.colorSwitch);
    }

    // write new force back to original unsorted location
    force += params.particleMass*params.gravity;
    newForce[index] += make_float4(force, 0.0f);

    // update particle color with respect to force or displacement
    float t;
    if (params.colorSwitch != 0)
    {
        if (params.colorSwitch == 1)
        {
            t = (1000.f*disp) / 100.f;
            if (t < -1.f) t = -1.f;
            if (t > 1.f)  t = 1.f;
            t += 1.f;
            t /= 2.f;
        }
        else
        {
            t = disp / 1.0f;
            if (t < 0.f) t = 0.f;
            if (t > 1.f)  t = 1.f;
        }
        float3 rgb = colorRamp(t);
        newColor[index] = make_float4( rgb, 0.5f );
    }
}



struct intShrink_functor
{
    float deltaTime;

    __host__ __device__
    intShrink_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile int    segData = thrust::get<3>(t);
        if (segData != params.contourSwitch) return;

        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        volatile float4 forData = thrust::get<2>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
        float3 force = make_float3(forData.x, forData.y, forData.z);

        // Explicit Euler
        vel += (force) * deltaTime;
        vel *= params.globalDamping;
        float3 ray = vel * deltaTime;
        //if (length(ray) > params.voxelSize.x)
        //    ray *= (params.voxelSize.x / length(ray));
        pos += ray;

        // set this to zero to disable collisions with cube sides
#if 1

        float3 worldSize = 0.5f * params.worldSize;
        if (pos.x > worldSize.x - params.particleRadius)
        {
            pos.x = worldSize.x - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -worldSize.x + params.particleRadius)
        {
            pos.x = -worldSize.x + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > worldSize.y - params.particleRadius)
        {
            pos.y = worldSize.y - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > worldSize.z - params.particleRadius)
        {
            pos.z = worldSize.z - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -worldSize.z + params.particleRadius)
        {
            pos.z = -worldSize.z + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

#endif

        if (pos.y < -worldSize.y + params.particleRadius)
        {
            pos.y = -worldSize.y + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};








#endif
