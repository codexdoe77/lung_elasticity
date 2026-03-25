 #ifndef CONSTANTS_H
#define CONSTANTS_H

namespace ParticleConstants{
#define ACCEL_GRAV      	0.0f      // m/s^2
#define DAMPING				0.99f		 // Unitless. Goes really slow if set below 0.5
#define TIMESTEP 			0.08f		 // seconds
#define VOXEL_SIZE      	0.08f        // meters
#define NUM_SPRINGS     	26			 // Number of spring connections between particles, excluding surface
#define NUM_PARTICLES   	32768        // default number of particles if program is launched with no inputs
#define ALPHA				0.5f         // Number b/t 0 and 1.0 that weights elasticity estimation with neighbors
#define SHEAR_MOD			40.f         // Theoretical Units: kPa
#define MAX_MOD         	90.f    	 // Theoretical Units: kPa
#define MIN_MOD         	0.1f   		 // Theoretical Units: kPa
#define ELASTIC_MOD     	20.f   		 // Theoretical Units: kPa
#define DISP_SCALE      	1000.f 		 // Scales underlying parameters from GPa to kPa and m to mm
#define ELAS_SCALE      	0.001f 	     // Scales underlying parameters from GPa to kPa and m to mm 0.0000001f
#define ERR_SCALE			1.f          // Scale error if necessary
#define UVW_SCALE			0.001f       // Scale UVW from voxel size to m
#define PARTICLE_TOL		0.0005f 	 // meters
#define CONVERGE        	80.0f        // Percent
#define STALL_TOL			0.0001f	     // kPa
#define ITERATION_BOUNDARY	1 	     // Number of timesteps boundary condition is implemented
#define ITERATION_INNER     2
#define ITERATION_EQ		3   // Number of timesteps after boundary condition the simulation is allowed to settle in equilibrium

}

#endif //CONSTANTS_H

