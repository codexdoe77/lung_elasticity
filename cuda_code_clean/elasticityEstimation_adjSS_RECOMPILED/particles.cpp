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
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_math.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <time.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <string>

#include "particleSystem.h"
#include "cmd_option.h"

//#define MAX_EPSILON_ERROR 5.00f   // remnant from CUDA particles code
#define THRESHOLD         0.25f     // defined for reading MR text files - no longer used
#define BONE_THRESH       150.f     // default bone threshold in HU
#define AIR_THRESH        -400.f    // default air threshold in HU

#define MAX_NUMBER_OF_TORUS 10   // maximum number of torus in the collider column

//char *FILE_MAG_OUT;
//char *FILE_ELAS_OUT;
//char *FILE_CONVERGE_OUT;
//char *FILE_ELASTICITY_OUT;

const uint width = 800, height = 800; // rendering window size
float air_thresh, bone_thresh;        // HU thresholds established to segment the ct input

char *FILE_MAG_OUT;
char *FILE_ELAS_OUT;
char *FILE_CONVERGE_OUT;
char *FILE_ELASTICITY_OUT;

DICOM_CT *input_ct;          // structure to hold all data from the input ct dicom
DICOM_STRUCT *structures;      // structure to hold all data from the input dicomrt structures
bool useInput = false;      // boolean to signal that a dicom ct was input
bool useContour = false;    // boolean to signal that a dicomrt structure was input
bool useOpenGL = false;
bool m_bAnchor = false;
char *UVW_file;
char *geom_file;
char *elas_file;
char *static_in;

dim3 cudaBlock(16, 16);
dim3 cudaGrid;
cudaExtent volumeSize = make_cudaExtent(128, 128, 128);
CHARDATA_VOLUME vdens1;

// booleans describing the current state of the simulation

bool UVW = false;
bool magnify = false;
bool tolFilter = false;
bool geom = false;
bool set_gravXY = false;
bool uvwON = false;
bool set_cum_tolerance = true;
bool set_initialGT = true;
bool set_initialRL = true;
bool deform = false;
bool initialIteration = true;
bool revertRL = false;
bool set_grav = true;
bool xy = false;
bool SA_iterations = false;
bool displayEnabled = true;
bool connectionSet = true;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
bool realTime = false;

uint colorSwitcher = 0;         // enumerator indicating which color map to render - anatomy / displacement / strain energy
uint contourSwitch = 0;         // enumerator indicating which contour is currently under control allowing rest length manipulation by the user
uint cntr_count = 0;            // total number of contours loaded by the user

uint userSize = 0;
uint osc_axis = 2;
uint oscillator = 0;
float amplitude = 0.f;

int idleCounter = 0;            // counts the frames without user interaction to trigger demo mode - currently disabled
int demoCounter = 0;            // counter for demo mode to switch the system intializiation state - cube / sphere / pyramid
const int idleDelay = 2000;     // number of idle frames before initializing demo mode

// Global Space Definers
uint numParticles = 0;          // total particles in the simulation
uint staticParticles = 0;       // number of static particles of the bony anatomy
uint dynamicParticles = 0;      // number of dynamic particles of soft tissue
float boundaryMultiplier = 0.00005; //Magnifies boundary pull accordingly. Default 0.00005.
uint3 gridSize;                 // the size of the global rendering space divided into voxels
uint3 dataSize;                 // the size of the input data volume in voxels
float3 voxelSize;               // the size of the voxels of the input data volume in mm
float3 worldSize;               // the total size of the global rendering space in mm
//int numIterations = 0; // run until exit

// simulation parameters
float timestep = TIMESTEP;  // seconds
float damping = DAMPING;    // unitless
float gravity = 0.0f;    // scaling factor (0 -> 2) x acceleration due to gravity
float grav[] = {0, 0, 0};   // gravity vector allwos gravity to rotate as the user changes views
//int iterations = 1

int ballRadius = 20;        // Radius of particle sphere in particles

// Particle-Particle collision data
float collideSpring = 0.f;
float collideDamping = 0.f;
float collideShear = 0.f;
float collideAttraction = 0.f;

// Theoretical Units shown, actual calculation performed in generic normalized grid units (-1,1) in each direction
// Hooke's Law
float springStrength = 0.20f;           // N/m
float springDamping  = 0.0f;
// Strain Energy Equation
float energyModulus = 0.f;            // GPa, 10^9 N/m^2
// Young & Shear Modulus Equation
float elasticModulus = ELASTIC_MOD;           // GPa, 10^9 N/m^2
float shearModulus   = SHEAR_MOD * ELAS_SCALE;           // GPa, 10^9 N/m^2
// Lennard-Jones Bi-Reciprocal Equation
float minPotential   = 0.f;
float stretch_x      = 0.f;
float stretch_y      = 0.f;

// Head Rotation Variables
float headLevel = 0.f;          // initial level above which the bony anatomy will rotate during headRotation - initialized by examing the mandible contour if possible
                                // can be adjusted in the slider bar menu
uint headSwitch = 0;            // indicates which axis to rotate about - x / y / z

uint profileSwitch = 0;         // indicates which data to dump during dumpProfiles - position / velocity / color

float *restLength;              // Multiplier to globally change the rest length of particle-particle connections: ranges from 0.5 to 2.5
float restLengthSlider = 1.f;   // variable to hold the user adjusted value in the slider bar menu

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

ParticleSystem *psystem = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;

const char *sSDKsample = "Anatomy Mass-Spring Simulation";

extern "C" void cudaInit(int argc, char **argv);

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*
Resets the rest length modifier of each contour to 1.
Launched when the system is reset using the keys '1','2','3','4','7','8'
*/
void resetRestLength()
{
    restLengthSlider = 1.f;
    uint count = 1;
    if (useContour)
        count += structures->CTRnumber;
    for (uint i=0; i<count; i++)
        restLength[i] = 1.f;

    psystem->resetRestLength();
}

// initialize particle system
void initParticleSystem(uint numParticles, uint staticParticles, uint dynamicParticles, uint3 dataSize, uint3 gridSize, float3 voxelSize, bool bUseOpenGL)
{

    printf(" Particle system is initialized %d\n", useInput); fflush(stdout);
    psystem = new ParticleSystem(numParticles, staticParticles, dynamicParticles, dataSize, gridSize, voxelSize, bUseOpenGL, useInput, useContour, input_ct->segment, &input_ct->dataset, structures, air_thresh, bone_thresh, headLevel, geom, geom_file,m_bAnchor);

    psystem->FILE_MAG_OUT = FILE_MAG_OUT;
    psystem->FILE_ELAS_OUT = FILE_ELAS_OUT;
    psystem->FILE_CONVERGE_OUT = FILE_CONVERGE_OUT;
    psystem->FILE_ELASTICITY_OUT = FILE_ELASTICITY_OUT;

    psystem->populateDataFromFile(&input_ct->dataset,FILE_MAG_OUT);
    if (useInput)
        {
            psystem->systemReset(ParticleSystem::CONFIG_INPUT, input_ct->dir, false, true, userSize, useInput, true);
        }

    psystem->resetTime();

    if (m_bAnchor)
    {
        psystem->inputStatic(static_in, m_bAnchor);
    }

    sdkCreateTimer(&timer);
    printf("\n Initialized! \n"); fflush(stdout);
}

void resetParticleSystem(uint numParticles, uint staticParticles, uint dynamicParticles, uint3 dataSize, uint3 gridSize, float3 voxelSize, bool bUseOpenGL)
{
    psystem->systemReset(ParticleSystem::CONFIG_POS_RESET, input_ct->dir, true, false, userSize, useInput, initialIteration);
    psystem->resetTime();

    sdkCreateTimer(&timer);
}

/*
Free allocated memory
*/
void cleanup()
{
    psystem->finalFree();
    if (useInput)
    {
        free(input_ct->segment);
        free(input_ct->dataset.array3D);
        if (useContour)
        {
            for (int c=0; c<structures->CTRnumber; c++)
            {
                if(structures->include[c] != 0)
                {
                    free(structures->contour[c].matrix);
                    free(structures->contour[c].CTRpoints);
                    free(structures->contour[c].lm_index);
                }
            }
            delete structures->contour;
            free(structures->include);
            free(structures->cntrParticles);
            free(structures->cntrSegData);
        }
        free(restLength);
    }

    sdkDeleteTimer(&timer);
}

/*
Computes the frames per second. Gives a measure of the calculation speed of the GPU
*/
void computeFPS(float iteration_counter)
{
    frameCount++;
    fpsCount++;


    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "elasticityEstimation (%d mass elements): %3.1f fps, Iteration Number : %4.1f", numParticles, ifps,iteration_counter);

        // glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}


/*
This display loop runs each iteration, it form the bulk of the glut main loop.
User input is transferred to the particle system class, and then to the GPU for simulation
calculations.
*/

float simTime = 0.f;                                //float, copied from particleSystem.cpp to represent simulation time.
uint iteration_counter=0;                           //Counts number of time steps
uint reset_counter=0;                               //Counter for iteration number (elasticity algorithm step)
float4 averages = make_float4(0.f,0.f, 0.f, 0.f);
float particle_tolerance = PARTICLE_TOL;            // cost threshold
float stalled_tolerance = STALL_TOL;
float last_tolerance = 0.f;
float new_tolerance = 0.f;
float cumulative_tolerance = 0;
float last_elastic = 0.f;
float best_elastic = 0.f;
float last_convergence = 0.f;
float best_convergence = 0.f;
char best_buffer[32];
char best_buffer_converge[32];
char first_buffer[32];
char init_buffer[32];
int last_guess;
int best_guess;
float new_error = 100000000;
float last_error = 100000000;
float best_error = 100000000;
uint loop_counter = 0;                                //Counts number of times through Simulated Annealing Loop, or number of stalls
uint counter_limit = 50;                          //Limit of number of times through Simulated Annealing Loop, or number of stalls
int step_size = 600;                      //step size
bool best = false;
float2 boundDisp = make_float2(0.f, 0.f);
float mmPercent = 0;
float best_mm = 0;

void resetSystemParameters()
{
    psystem->systemReset(ParticleSystem::CONFIG_INPUT, input_ct->dir, true, false, userSize, useInput, false);

    set_cum_tolerance = false;

    iteration_counter = 0;

    gravity = 0.f;
    set_grav = true;
    psystem->setGravity(0,0,0);
    damping = DAMPING;
}
void setCumTolerance()
{
    cumulative_tolerance = particle_tolerance * numParticles * 0.01f;
    printf("\n Tolerance: %4.3f (%4.6f)",cumulative_tolerance,particle_tolerance);
}

void checkLastError()
{
    if (last_error < best_error) // || last_convergence > best_convergence)
    {
        best_error = last_error;
        best_guess = last_guess;
        best_elastic = last_elastic;
        best_convergence = last_convergence;
        best_mm = mmPercent;
        time_t rawtime;
        struct tm * timeinfo;
        time (&rawtime);
        timeinfo = localtime (&rawtime);
        strftime (best_buffer,32,"%m.%d.%y-%H.%M.%S",timeinfo);
        printf("\n ||| BEST: %4.3f kPa -> %4.3f m||| \n",best_elastic,best_error);
        // psystem->dumpResults(&input_ct->dataset);
    }
        // psystem->dumpResults(&input_ct->dataset);
}
void display_checkConvergence()
{

    if (set_cum_tolerance)
        setCumTolerance();
    if (!bPause)
    {
        psystem->resetElasticityforOptimization(false);
        psystem->setGravity(0.f, 0.f, 0.f);

        initialIteration = false;;

        averages = psystem->updateElasticity(reset_counter,particle_tolerance,input_ct->dataset.array3D, m_bAnchor, &input_ct->dataset, mmPercent);
        // averages = psystem->updateElasticityOld(reset_counter,particle_tolerance,input_ct->dataset.array3D, m_bAnchor, &input_ct->dataset);

        float elastic_change = fabs(averages.y - last_elastic);
        last_error = fabs(averages.x);
        last_elastic = averages.y;
        last_convergence = averages.z;


        checkLastError();
        psystem->dumpResults(&input_ct->dataset);

        if  (fabs(last_error) < cumulative_tolerance)
        {
            printf("\n Error Convergence Reached at Iteration %d",reset_counter);

            resetSystemParameters();

            printf("\n\n %d. ITERATION LOOP\n\n",loop_counter);


            reset_counter=0;
            psystem->resetElasticityforOptimization(true);

            averages = make_float4(0.f,0.f, 0.f, 0.f);
            last_elastic = 0.f;

            loop_counter++;

        }
        else if (elastic_change < stalled_tolerance)
        {
            printf("\n Solver stalled at Iteration %d",reset_counter);

            resetSystemParameters();

            // // 🚨 Check for NaNs introduced by systemReset // add 5-10-25
            // float* pos = psystem->getCurrentPositions();
            // for (int i = 0; i < psystem->getDynamicParticles(); i++) {
            //     if (std::isnan(pos[4*i+0]) || std::isnan(pos[4*i+1]) || std::isnan(pos[4*i+2])) {
            //         printf("❌ NaN in m_hPos AFTER resetSystemParameters at particle %d\n", i);
            //         printf("    Position: (%.4f, %.4f, %.4f)\n", pos[4*i+0], pos[4*i+1], pos[4*i+2]);
            //         exit(EXIT_FAILURE);
            //     }
            // }             // end 5-10-25
            reset_counter=0;
            // psystem->resetElasticityforOptimization(true);
            psystem->resetElasticityNeighbors(&input_ct->dataset);
            averages = make_float4(0.f,0.f, 0.f, 0.f);
            last_elastic = 0.f;
            last_convergence = 0.f;

            printf("\n\n %d. ITERATION LOOP\n\n",loop_counter);

            loop_counter++;
        }
       else
        {
            resetSystemParameters();
            reset_counter++;
            // if (reset_counter == 1)
            // {
            //     exit(0);
            // }

        }
    }



}

void display_printConvergence()
{
    //exit if convergence occurs
    if (best_error < cumulative_tolerance && best_convergence > CONVERGE)
    {
        psystem->dumpResults(&input_ct->dataset);
        printf("\n\n ---Results: ----\n \
        Cumulative Error: %4.3f \n \
        Best Guess - Elastic Modulus: %4.9f \n \
        Ground Truth - File Name: %s \n \
        Best Guess Error - File Name: %s  \n \
        Convergence: %4.3f %%\n", \
        best_error,best_elastic,first_buffer,best_buffer,best_convergence);

        char outpath[255];
        sprintf(outpath,"%s/elasticResults.txt",FILE_ELASTICITY_OUT);
        std::ofstream output_results;
        output_results.open(outpath, std::ofstream::out | std::ofstream::app);

        output_results << "Ground Truth Time"   << ": " << first_buffer << std::endl;
        output_results << "Best Results Time"   << ": " << best_buffer << std::endl;
        output_results << "Iteration Number"    << ": " << reset_counter << std::endl;
        output_results << "Best Guess"    << ": " << best_guess << std::endl;
        output_results << "Cumulative Error"    << ": " << best_error << std::endl;
        output_results << "YM Average"          << ": " << best_elastic<< std::endl;
        output_results << "Convergence"         << ": " << best_convergence << std::endl;
        output_results << "MM Convergence"      << ": " << best_mm << std::endl;

        output_results << std::endl;

        output_results.close();

        exit(EXIT_SUCCESS);
    }
    //alternative exit
    // if (reset_counter == 2 || loop_counter == 30 )
    /*
    if (loop_counter == 30 )
    {
        psystem->dumpResults(&input_ct->dataset);

        printf("\n\n ---Results: ----\n \
        Cumulative Error: %4.3f \n \
        Best Guess - Elastic Modulus: %4.9f \n \
        Ground Truth - File Name: %s \n \
        Best Guess Error - File Name: %s  \n \
        Convergence: %4.3f %%\n", \
        best_error,best_elastic,first_buffer,best_buffer,best_convergence);


        char outpath[255];
        sprintf(outpath,"%s/elasticResults.txt",FILE_ELASTICITY_OUT);
        std::ofstream output_results;
        output_results.open(outpath, std::ofstream::out | std::ofstream::app);

        output_results << "Ground Truth Time"   << ": " << first_buffer << std::endl;
        output_results << "Best Results Time"   << ": " << best_buffer << std::endl;
        output_results << "Iteration Number"    << ": " << reset_counter << std::endl;
        output_results << "Best Guess"    << ": " << best_guess << std::endl;
        output_results << "Cumulative Error"    << ": " << best_error << std::endl;
        output_results << "YM Average"          << ": " << best_elastic<< std::endl;
        output_results << "Convergence"         << ": " << best_convergence << std::endl;
        output_results << "MM Convergence"      << ": " << best_mm << std::endl;
        output_results << std::endl;

        output_results.close();

       exit(EXIT_SUCCESS);
    }
*/

}


void display_UpdateParticleSystemGlobalParams()
{
    sdkStartTimer(&timer);
    float worldMax = std::max( worldSize.x, std::max( worldSize.y, worldSize.z) );

if (!bPause)
    {

        if (set_initialRL)
        {
            psystem->recordRestLengths(deform, set_initialRL);
            set_initialRL = false;
        }
        psystem->setDamping(damping);
        if (set_grav)
        {
            grav[0] = 0*gravity * ACCEL_GRAV;//gravity * ACCEL_GRAV; // * sin(camera_rot_lag[0]*PI/180) * sin(camera_rot_lag[1]*PI/180) / (0.5f * gridSize.x * voxelSize.x);
            grav[1] = 0*gravity * ACCEL_GRAV; //gravity * ACCEL_GRAV; // * cos(camera_rot_lag[0]*PI/180) / (0.5f * gridSize.y * voxelSize.y);
            grav[2] = 1*gravity * ACCEL_GRAV; // * sin(camera_rot_lag[0]*PI/180) * cos(camera_rot_lag[1]*PI/180) / (0.5f * gridSize.z * voxelSize.z);
            psystem->setGravity(grav[0],grav[1],grav[2]);
        }
        if (set_gravXY)
        {
            grav[0] = 1*gravity * ACCEL_GRAV;
            grav[1] = 1*gravity * ACCEL_GRAV;
            grav[2] = 0*gravity * ACCEL_GRAV;
            psystem->setGravity(grav[0],grav[1],grav[2]);
        }

        psystem->setCollideSpring(collideSpring);
        psystem->setCollideDamping(collideDamping);
        psystem->setCollideShear(collideShear);
        psystem->setCollideAttraction(collideAttraction);

        psystem->setSpringConstant(springStrength);
        psystem->setSpringDamping(springDamping);

        psystem->setShearModulus(shearModulus);

        psystem->setMinimumPotential(minPotential);
        psystem->setStretchiness(stretch_x,stretch_y);

        restLengthSlider = 1.f;
        restLength[contourSwitch] = restLengthSlider;
        psystem->setRestLength(restLength[contourSwitch]);
        // psystem->setColliderOuterRadius(colliderOuterRadius, worldMax);
        // psystem->setColliderInnerRadius(colliderInnerRadius, worldMax);
        // psystem->setColliderLength(colliderLength);
        // psystem->setHeadLevel(headLevel);
        psystem->accumulateTime(timestep);

        simTime = psystem->update(timestep, magnify, iteration_counter);

    }

}

void display_UpdateParticleSystemSimulation()
{
    if (iteration_counter == 0)
    {

        psystem->recordRestLengths(deform, initialIteration);
        psystem->resetElasticityforOptimization(false);

        gravity = 0;
        set_grav = true;
        damping = DAMPING;

    }
    else
    {
        if (iteration_counter <= ITERATION_BC)
        {
            psystem->moveBoundaryToTarget(&input_ct->dataset, iteration_counter);
            // simTime = psystem->update(timestep, magnify, iteration_counter);
        }
        if (iteration_counter ==  ITERATION_EQ)
        {

            deform = true;
            psystem->setPreMagPos();
            psystem->recordRestLengths(deform, initialIteration);
            //psystem->resetElasticityforMagnify(ELASTIC_MAGNIFY); //**CORRECT Line

            resetParticleSystem(numParticles, staticParticles, dynamicParticles, dataSize, gridSize, voxelSize, useOpenGL);

            gravity = 1.f;
            set_grav = true;
            damping = 0.99f;
            deform = false;
            magnify = true;
        }

    }
}

void display()
{
    iteration_counter++;
    if (iteration_counter >= ITERATION_MAGNIFY)
    {
        display_checkConvergence();
        display_printConvergence();
    }

    display_UpdateParticleSystemGlobalParams();
    display_UpdateParticleSystemSimulation();
    computeFPS(iteration_counter);
}




inline float frand()
{
    return rand() / (float) RAND_MAX;
}


void printCommandLineHelp()
{
  printf("\n Command Line Flags:");
            printf("\n    -help:     Display all flag options.");
            printf("\n    -n:        Set number of particles for generic system. Default = 32678.");
            printf("\n    -grid:     Set sizeof cubic grid. Default = 96.");
            printf("\n    -input:    Prompts a file chooser window to select data folder containing DICOM or text files for input data set.");
            printf("\n               Text files require a parameter file within the data folder containing image dimensions and voxel size in mm.");
            printf("\n               Overrides -n and -grid flags.");
            printf("\n    -ct:       Modifier of -input flag. Signals program to use air and bone thresholds to fill dataset. Default values set at -400 and 150 HU. ");
            printf("\n               Other input (i.e. MR) can be thresholded through the parameter file, or will employ a binary default threshold of 0.");
            printf("\n    -athresh:  Modifier of -input flag. Manually set the threshold of air for CT input. Default is -400 HU.");
            printf("\n               Voxels below this value will not be included in the simulation data set.");
            printf("\n    -bthresh:  Modifier of -input flag. Manually set the threshold of bone for CT input. Default is 150 HU.");
            printf("\n               Voxels above this value are considered fixed points during simulation.");
            printf("\n    -body:     Modifier of -struct flag. Automatically load body contour and eliminate any voxels outside of body contour from the simulation data set.");
            printf("\n    -parse:    Modifier of -struct flag. Similarly remove all voxels not contained within the selected array of contours.");
            printf("\n               This option currently does not remove the skeletal structure, however.\n");
            printf("\n  -downSample  Set the factor to down sample the x-y directions. (-downSample=2 will reduce a 512x512 image to 256x256)\n");
}


void checkandloadInputCT(int argc, char **argv)
{
      float inputThreshold = THRESHOLD;
  if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {

            getCmdLineArgumentString( argc, (const char**)argv, "input", &input_ct->dir );
            printf("\n Input Path: %s",input_ct->dir);
            fflush(stdout);


            // check for the flag to down-sample the input data
            int downSamp = 1;
            if(checkCmdLineFlag(argc, (const char**)argv, "downSample"))
                downSamp = getCmdLineArgumentInt(argc, (const char **)argv, "downSample");
            bool zdown = false;
            if(checkCmdLineFlag(argc, (const char**)argv, "zdown"))
                zdown = true;

            char *params_file;
            if (checkCmdLineFlag(argc, (const char **) argv, "params"))
            {
                getCmdLineArgumentString( argc, (const char**)argv, "params" ,&params_file );
                printf("\n Parameter File: %s",params_file);fflush(stdout);
            }

            if (checkCmdLineFlag(argc, (const char **) argv, "geometry"))
            {
                getCmdLineArgumentString( argc, (const char**)argv, "geometry" ,&geom_file );
                printf("\n Geometry File: %s",geom_file);
                geom = true;
            }
            if (checkCmdLineFlag(argc, (const char **) argv, "elasFile"))
            {
                getCmdLineArgumentString( argc, (const char**)argv, "elasFile" ,&elas_file );
                printf("\n Elasticity File: %s",elas_file);
            }
            if (checkCmdLineFlag(argc, (const char **) argv, "static"))
            {
                getCmdLineArgumentString( argc, (const char**)argv, "static" ,&static_in );
                printf("\n Static Row: %s",static_in);fflush(stdout);
                m_bAnchor = true;
            }

            if (checkCmdLineFlag(argc, (const char **) argv, "xy"))
            {
                xy = true;
                if (xy)
                {
                    set_gravXY = true;
                    printf("\n Evaluating XY Perturbation Force");
                }
            }
            if (checkCmdLineFlag(argc, (const char **) argv, "boundaryMultiplier"))
            {
                boundaryMultiplier = getCmdLineArgumentFloat(argc, (const char **)argv, "boundaryMultiplier");
                printf("\n Boundary Multiplier: %4.6f", boundaryMultiplier);
            }

            //Load DICOM files from input directory
            load_files(input_ct->dir,&input_ct->files,&input_ct->dataset,input_ct->type,input_ct->date,downSamp);
            printf("\n Data is type %s ",input_ct->type); fflush(stdout);



            // if the input data is in text file format, search for a parameter file defining the array size and voxel size
            int y = strncmp(input_ct->type,"txt",3);

            bool ytrue = false;
            if (y == 0)
            {
                ytrue = true;
            }
            if ( ytrue )
            {
                char name[255];
                sprintf(name, "%s", params_file);

                FILE *fp;
                fp = fopen(name, "r");

                if (fp != NULL)
                    {
                        fscanf(fp,"%d %d %d %f %f %f %f",&dataSize.x,&dataSize.y,&dataSize.z,&voxelSize.x,&voxelSize.y,&voxelSize.z,&inputThreshold);
                    }
                else
                    {
                        printf("LOAD PARAMETER FILE ERROR \n");
                    }



                dataSize.x /= downSamp;
                dataSize.y /= downSamp;
                dataSize.z /= downSamp;
                input_ct->dataset.params.arraySize.x = dataSize.x;
                input_ct->dataset.params.arraySize.y = dataSize.y;
                input_ct->dataset.params.arraySize.z = dataSize.z;
                if (zdown) dataSize.z /= downSamp;
                voxelSize.x /= 1000.f;
                voxelSize.y /= 1000.f;
                voxelSize.z /= 1000.f;
                input_ct->dataset.params.voxelSize.x = voxelSize.x;
                input_ct->dataset.params.voxelSize.y = voxelSize.y;
                input_ct->dataset.params.voxelSize.z = voxelSize.z;
                fclose(fp);



            }
            else
            {
                // assuming DICOM format, assign the array size and voxel size to the global variables
                dataSize.x = input_ct->dataset.params.arraySize.x / downSamp;
                dataSize.y = input_ct->dataset.params.arraySize.y / downSamp;
                dataSize.z = input_ct->dataset.params.arraySize.z;
                if (zdown) dataSize.z/=downSamp;
                voxelSize.x = (input_ct->dataset.params.voxelSize.x * (float)downSamp) / 1000.f;
                voxelSize.y = (input_ct->dataset.params.voxelSize.y * (float)downSamp) / 1000.f;
                voxelSize.z = input_ct->dataset.params.voxelSize.z / 1000.f;
                if (zdown) voxelSize.z *= (float)downSamp;
            }

            // load the pixel data from the DICOM files
            load_data(input_ct->dir,&input_ct->files,&input_ct->dataset,input_ct->type,downSamp,zdown);


            // adjust the parameters for down-sampling
            input_ct->dataset.params.arraySize.x /= downSamp;
            input_ct->dataset.params.arraySize.y /= downSamp;
            input_ct->dataset.params.voxelSize.x *= downSamp;
            input_ct->dataset.params.voxelSize.y *= downSamp;
            if (zdown)
            {
                input_ct->dataset.params.arraySize.z /= downSamp;
                input_ct->dataset.params.voxelSize.z *= downSamp;
            }

            // initialize the particle counter to zero
            numParticles = 0;
            dynamicParticles = 0;
            staticParticles = 0;

            // establish the size of the global rendering space grid at 20 units larger than the array size
            gridSize.x = 20 + dataSize.x;
            gridSize.y = 20 + dataSize.y;
            gridSize.z = 20 + dataSize.z;


            // allocate and initialize the enumerator array
            input_ct->segment = (int*)malloc( dataSize.x * dataSize.y * dataSize.z * sizeof(int) );

            memset( input_ct->segment, 0, dataSize.x * dataSize.y * dataSize.z * sizeof(int) );

            // set the air and bone thresholds to default value, then check for command line flags
            air_thresh = AIR_THRESH;
            bone_thresh = BONE_THRESH;
            if (checkCmdLineFlag(argc, (const char **)argv, "bthresh"))
                bone_thresh = getCmdLineArgumentFloat(argc, (const char **)argv, "bthresh");
            if (checkCmdLineFlag(argc, (const char **)argv, "athresh"))
                air_thresh = getCmdLineArgumentFloat(argc, (const char **)argv, "athresh");

            // check for the CT command line flag.
            // if found, this will apply the air and bone thresholds to the data.
            // values below the air threshold will not be rendered, value above the bone threshold will be considered static

            if (checkCmdLineFlag(argc, (const char **)argv, "ct"))
            {
                printf("\n Soft Tissue: %3.2f -> %3.2f\n",air_thresh,bone_thresh); fflush(stdout);
                char outpath[255];
                sprintf(outpath,"%s/Array.txt",FILE_ELASTICITY_OUT);

                std::ofstream output_Array;
                output_Array.open(outpath);
                // check for command line flags indicating that only bone or only soft tissue is to be rendered
                uint check = 0;
                if (checkCmdLineFlag(argc, (const char **)argv, "bonly")) check++;
                if (checkCmdLineFlag(argc, (const char **)argv, "sonly")) check--;

                for (uint n=0; n<dataSize.x*dataSize.y*dataSize.z; n++)
                {
                    // add both bone and soft tissue to the particle array for rendering
                    if (check==0)
                    {
                        if (input_ct->dataset.array3D[n] > bone_thresh)
                        {
                            numParticles++;
                            staticParticles++;
                            input_ct->segment[n] = -1000;
                        }
                        else if (input_ct->dataset.array3D[n] > air_thresh)
                        {
                            numParticles++;
                            dynamicParticles++;
                            input_ct->segment[n] = 1000;
                        }
                        else input_ct->segment[n] = 0;
                    }
                    // add only bone to the particle array for rendering
                    else if (check==1)
                    {
                        if (input_ct->dataset.array3D[n] > bone_thresh)
                        {
                            numParticles++;
                            staticParticles++;
                            input_ct->segment[n] = -1000;
                        }
                        else input_ct->segment[n] = 0;
                    }
                    // add only soft tissue to the particle array for rendering
                    else
                    {

                        // if (input_ct->dataset.array3D[n] < -600 && input_ct->dataset.array3D[n] > -800) //for lung data
                        if (input_ct->dataset.array3D[n] != 0) // && input_ct->dataset.array3D[n] > -1100) //for lung data
                        // if (input_ct->dataset.array3D[n] < bone_thresh && input_ct->dataset.array3D[n] > air_thresh)
                        {
                            output_Array << n << ", " << input_ct->dataset.array3D[n] << std::endl;
                            (n == dataSize.x * dataSize.y * dataSize.z - 1) ? output_Array << std::endl : output_Array << " ";
                            numParticles++;
                            dynamicParticles++;
                            input_ct->segment[n] = 1000;

                        }
                        else input_ct->segment[n] = 0;
                    }
                }
                output_Array.close();
            }

            // particle array for rendering

            // else
            // {

            // if (checkCmdLineFlag(argc, (const char **)argv, "sphere"))
            //     {
            //     std::cout <<" Loading the data into input \n";
            //        std::ofstream output_Sphere;
            //        int *r;
            //        r = new int[dataSize.x * dataSize.y * dataSize.z];
            //        uint n = 0;
            //           for (int i = 0; i < 10; ++i)
            //             {
            //                 for (int j = -10; j < 10; ++j)
            //                 {
            //                     for (int k = -10; k < 10; ++k)
            //                     {

            //                         r[n] = i*i + j*j + k*k;
            //                         if (r[n] < 400)
            //                         // if (r[n] < 4096)
            //                         {
            //                             input_ct->dataset.array3D[n] = -450.f;
            //                         }

            //                         (n == dataSize.x * dataSize.y * dataSize.z - 1) ? output_Sphere << std::endl : output_Sphere << " ";
            //                         n++;
            //                     }
            //                 }
            //             }


            //             output_Sphere.close();
            //       }


            // if (checkCmdLineFlag(argc, (const char **)argv, "cube"))
            //     {

            //         int n = 0;
            //         for (int i = 0; i < 25; ++i)
            //         {
            //             input_ct->dataset.array3D[n] = -450.f;
            //         }

            //         n++;

            //     }
            //     for (uint n=0; n<dataSize.x*dataSize.y*dataSize.z; n++)
            //         if (input_ct->dataset.array3D[n] < inputThreshold)
            //         {
            //             numParticles++;
            //             dynamicParticles++;
            //             input_ct->segment[n] = 1000;
            //         }
            // }

            // if no RTSTRUCT is defined, create a single global rest length multiplier
            restLength = (float*)malloc(sizeof(float));
            restLength[0] = 1.f;
            
            // set the input boolean to true
            useInput = true;

    }
}

void checkandloadOutputDirectories(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **) argv, "FILE_MAG_OUT"))
    {
        getCmdLineArgumentString( argc, (const char**)argv, "FILE_MAG_OUT" ,&FILE_MAG_OUT );
        //std::cout << "FILE_MAG_OUT: " << FILE_MAG_OUT << std::endl;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "FILE_ELAS_OUT"))
    {
        getCmdLineArgumentString( argc, (const char**)argv, "FILE_ELAS_OUT" ,&FILE_ELAS_OUT );
        //std::cout << "FILE_ELAS_OUT: " << FILE_ELAS_OUT << std::endl;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "FILE_CONVERGE_OUT"))
    {
        getCmdLineArgumentString( argc, (const char**)argv, "FILE_CONVERGE_OUT" ,&FILE_CONVERGE_OUT );
        //std::cout << "FILE_CONVERGE_OUT: " << FILE_CONVERGE_OUT << std::endl;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "FILE_ELASTICITY_OUT"))
    {
        getCmdLineArgumentString( argc, (const char**)argv, "FILE_ELASTICITY_OUT" ,&FILE_ELASTICITY_OUT );
        //std::cout << "FILE_ELASTICITY_OUT: " << FILE_ELASTICITY_OUT << std::endl;
    }
}


void initContourSegmentData()
{
  if (useInput && useContour)
    {
        int count2 = structures->cntr_count;
        for ( int i=0; i<2*structures->cntr_count; i++)
        {
            for( int j=1; j<count2; j++)
            {
                if ( structures->cntrSegData[j-1] > structures->cntrSegData[j] )
                {
                    int tempint = structures->cntrSegData[j];
                    structures->cntrSegData[j] = structures->cntrSegData[j-1];
                    structures->cntrSegData[j-1] = tempint;
                }
            }
            count2--;
        }
        printf("\n Contour Segment Data - Sorted:  ");
        for (int i=0; i<structures->cntr_count; i++)
                printf(" %d ",structures->cntrSegData[i]);
    }

}


void printParticlesStatus()
{
    printf("\n voxel (m): %2.2g x %2.2g x %2.2g", voxelSize.x, voxelSize.y, voxelSize.z ); fflush(stdout);
    printf("\n data: %d x %d x %d = %d voxels", dataSize.x, dataSize.y, dataSize.z, dataSize.x*dataSize.y*dataSize.z);
    printf("\n grid: %d x %d x %d = %d cells", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z); fflush(stdout);
    printf("\n Total Particles: %d", numParticles); fflush(stdout);
    printf("\n Static Particles: %d", staticParticles); fflush(stdout);
    printf("\n Dynamic Particles: %d\n", dynamicParticles); fflush(stdout);
}
void checkandsetGrid(int argc, char ** argv)
{
  if (checkCmdLineFlag(argc, (const char **) argv, "grid"))
            {
                uint gridDim = getCmdLineArgumentInt(argc, (const char **) argv, "grid");
                gridSize.x = gridSize.y = gridSize.z = gridDim;
            }

}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    printf("%s Starting...\n", sSDKsample);

    numParticles = NUM_PARTICLES;
    staticParticles = 0;
    dynamicParticles = NUM_PARTICLES;
    float voxelDim = VOXEL_SIZE;
    dataSize.x = dataSize.y = dataSize.z = 0;
    voxelSize.x = voxelSize.y = voxelSize.z = voxelDim;
    float dim = pow( (float)numParticles, (1.f/3.f) ); //length of a side of the particle cube
    gridSize.x = gridSize.y = gridSize.z = 3 * (uint)dim; //set grid size to twice the size of the particle cube
   //set grid size to twice the size of the particle cube
    //numIterations = 0;
    float inputThreshold = THRESHOLD;

    cudaGrid = dim3(iDivUp(width, cudaBlock.x), iDivUp(height, cudaBlock.y));

    cudaInit(argc,argv);

    input_ct = new DICOM_CT;

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char**)argv, "help"))
        {
            printCommandLineHelp(); return SUCCESS;
        }

            // Get declared input directory
        checkandloadInputCT(argc, argv);
        checkandloadOutputDirectories(argc, argv);

        // create a single global rest length multiplier
        restLength = (float*)malloc(sizeof(float));
        restLength[0] = 1.f;

        if (checkCmdLineFlag(argc, (const char **) argv, "SA_iterations"))
            SA_iterations = true;

        if (checkCmdLineFlag(argc, (const char **) argv, "tolFilter"))
            tolFilter = true;

        if (checkCmdLineFlag(argc, (const char **) argv, "UVW"))
        {
            UVW = true;
            uvwON = true;
            getCmdLineArgumentString( argc, (const char**)argv, "UVW" ,&UVW_file );
            printf(" UVW: %s\n", UVW_file );
            fflush(stdout);
        }

    }
    else
    {
        // if there are no command line arguments
        // create a single global rest length multiplier
        restLength = (float*)malloc(sizeof(float));
        restLength[0] = 1.f;
    }

    // Command line parsing completed

    initContourSegmentData();
    printParticlesStatus();

    // initalize the particle system, parameters, and menus
    initParticleSystem(numParticles, staticParticles, dynamicParticles, dataSize, gridSize, voxelSize, useOpenGL);

    // begin the main glut loop
    printf("\n running without rendering...\n");
    while (true) {
        display();
    }

    if (psystem)
        delete psystem;
    if (input_ct)
        delete input_ct;
    if (structures)
        delete structures;

    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
