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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define FETCH(t, i) t[i]

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    float3 comPos;          // center of mass of particle system

  // Collider Apparatus Definition
    float3 colliderPos;                 // position
    float3 colliderVel;                 // velocity
    float3 colliderOrientation;         // orientation vector
    float  colliderOuterRadius[10];     // array of torus outer radii, maximum 10
    float  colliderInnerRadius[10];     // array of torus inner radii, maximum 10
    float  colliderMass;                // mass
    uint   colliderLength;              // number of torus, 1-10
    uint   colliderControl;             // enumerator to indicate which torus the user can currently adjust the radii of

  // Global Variables
    float3 gravity;                 // gravity factor
    float  globalDamping;           // global damping factor
    float  particleRadius;          // particle radius
    float  particleSpacing;         // particle spacing for geometric objects
    float  particleMass;            // particle mass

  // Rigid Body Variables - not currently in use
  //  float  rigidMass;
  //  float  rigidRatio;
  //  float  elastRatio;
  //  float3 rigidTensor[3];

    float restLengthMult;

  // Hooke's Law Force Variables
    float springStrength;
    float springDamping;

  // Elastic & Shear Modulus Variables
    float MINmod;
    float MAXmod;
    float elasticModulus;
    float shearModulus;

  // Strain Energy Variables
    float energyModulus;

  // Lennard-Jones Bi-Reciprocal Function Variables
    float  minPotential;
    float2 stretchiness;

  // Color variables for rendering
    uint colorSwitch;
    uint contourSwitch;

  // Global Space Definition Variables
    uint3  gridSize;        // number of "voxels" in each direction of the global space
    float3 voxelSize;       // size of each voxel in mm
    float3 worldSize;       // size of global space in mm
    uint   numCells;        // total number of voxels in the global space
    float3 worldOrigin;     // set origin (negative most voxel in x,y,z) so that the center of the global space is (0,0,0)

  // Particle-Particle Collision Variables
    float spring;           // user set elastic spring factor
    float damping;          // user set spring damping factor
    float shear;            // user set shear factor
    float attraction;       // user set attractive factor
    float boundaryDamping;  // user set boundary damping factor

  // Head Rotation Variables
    uint headSwitch;        // enumerator taht switches axis of rotation
    uint headStart;         // id of threshold particle to be included in rotation
    float3 headAngle;       // holds current angle of rotation in each direction
    float headLevel;        // user adjusted height above which to include in rotation

    uint oscillator;
    float3 osc_axis;
    float3 osc_origin;
    float amplitude;
    float simulationTime;
};

#endif
