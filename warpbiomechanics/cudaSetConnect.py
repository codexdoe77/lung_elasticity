import numpy as np
import ctypes
import os
import sys
from cuda import cuda, cudart
from py_cuda_helper import checkCudaErrors, findCudaDevice, KernelHelper
import open3d as o3d
from pxr import Usd, UsdGeom, Sdf


class SetConnect:
	def __init__(self, arraySize, iso_resolution, volume ):
		# iso_resolution should be passed in units of mm
		
		self.devID = findCudaDevice()
		self.deviceProps = checkCudaErrors(cudart.cudaGetDeviceProperties(self.devID))
		print("CUDA device [{}] has {} Multi-Processors SM {}.{}".format(self.deviceProps.name,
																			self.deviceProps.multiProcessorCount,
																			self.deviceProps.major,
																			self.deviceProps.minor))
		if (self.deviceProps.major < 2):
			print("Requires SM 2.0 or higher for support of Texture Arrays.  Script will exit...")
			sys.exit()
		
		self.numSprings = 26
		self.totalSprings = 0
		self.numGridCells = arraySize[0] * arraySize[1] * arraySize[2]

		self.binary_volume = volume
		self.index_volume = np.zeros_like(self.binary_volume, dtype=np.uint32)

		# print(self.numGridCells)
		self.numParticles = np.count_nonzero(volume)

		self.worldOrigin = np.zeros(3, dtype=np.float32)

		self.arraySize = arraySize
		self.resolution = np.full(3, iso_resolution, dtype=np.float32)
		self.particle_radius = 0.5 * iso_resolution # mm
		self.particle_volume = (iso_resolution ** 3) # mm^3
		self.density = 1
		self.particle_mass = self.density * self.particle_volume

		self.hPos = np.zeros(4*self.numParticles, dtype=np.float32)
		self.nzindices = (np.nonzero(volume)[0])
		# print(self.nzindices)

		self.hConnections = None 
		self.hRestLengths = None 
		self.hLineSet = None
		self.hconStart = np.zeros(self.numParticles, dtype=np.uint32)
		self.hconCount = np.zeros(self.numParticles, dtype=np.uint32)
		self.hSurface = np.zeros(self.numParticles, dtype=np.float32)
		self.hSurfVolume = np.zeros(self.arraySize[0]*self.arraySize[1]*self.arraySize[2], dtype=np.float32)
		self.hNormals = None
		self.surfParticles = 0

		self.faces = {}
		self.triangles = []
		self.tetras = []
		self.tetvols = []
		self.tet_centroids = []
		self.totalVolume = 0.0
		self.edges = []

		print(self.worldOrigin, ' World Origin')
		print(self.arraySize, " Grid Dimensions")
		print(self.resolution, " Resolution")
		print(self.numParticles, " Total Particles/Vertices")
		
		self.setConnectKernels = '''
			#define NUM_SPRINGS 26
			#define SPRING_SEARCH_SIZE 3
			typedef unsigned int uint;

			__device__ float4 index2float4(uint index, int3 arraySize, float3 voxelSize)
			{
				int page = (arraySize.x * arraySize.y);
				float Z = floor(__int2float_rn(index / page)) * voxelSize.z;
				float Y = floor(__int2float_rn((index % page) / arraySize.x)) * voxelSize.y;
				float X = floor(__int2float_rn((index % page) % arraySize.x)) * voxelSize.x;
				return make_float4(X, Y, Z, 0.0);
			}

			// calculate position in uniform grid
			__device__ int3 calcGridPos(float3 p, float3 worldOrigin, float3 voxelSize)
			{
				int3 gridPos;
				gridPos.x = floor((p.x - worldOrigin.x) / voxelSize.x);
				gridPos.y = floor((p.y - worldOrigin.y) / voxelSize.y);
				gridPos.z = floor((p.z - worldOrigin.z) / voxelSize.z);
				return gridPos;
			}

			// calculate address in grid from position (clamping to edges)
			__device__ uint calcGridHash(int3 gridPos, int3 gridSize)
			{
				gridPos.x = gridPos.x % (gridSize.x-1);  // wrap grid, assumes size is power of 2
				gridPos.y = gridPos.y % (gridSize.y-1);
				gridPos.z = gridPos.z % (gridSize.z-1);
				return __umul24(__umul24(gridPos.z, gridSize.y), gridSize.x) + __umul24(gridPos.y, gridSize.x) + gridPos.x;
			}

			extern "C" __global__
			void initPosD( 	float4 *pos,
							uint   *indices,
							uint   *volume, 
							uint    numParticles,
							int     gridSizex,
							int     gridSizey,
							int     gridSizez,
							float   voxelSizex,
							float   voxelSizey,
							float   voxelSizez)
			{
				uint p = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

				if (p >= numParticles) return;

				uint index = indices[p];

				pos[p] = index2float4(index, make_int3(gridSizex, gridSizey, gridSizez), make_float3(voxelSizex, voxelSizey, voxelSizez));
				volume[index] = p;
			}

			extern "C" __global__ 
			void calcHashD( uint   *gridParticleHash,  // output
							uint   *gridParticleIndex, // output
							float4 *pos,               // input: positions
							uint    numParticles,
							int     gridSizex,
							int     gridSizey,
							int     gridSizez,
							float   worldOriginx,
							float   worldOriginy,
							float   worldOriginz,
							float   voxelSizex,
							float   voxelSizey,
							float   voxelSizez)
			{
				uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

				if (index >= numParticles) return;

				volatile float4 p = pos[index];

				// get address in grid
				int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z), make_float3(worldOriginx, worldOriginy, worldOriginz), make_float3(voxelSizex, voxelSizey, voxelSizez));
				uint hash = calcGridHash(gridPos, make_int3(gridSizex, gridSizey, gridSizez));

				// store grid hash and particle index
				gridParticleHash[index] = hash;
				gridParticleIndex[index] = index;
			}

			extern "C" __global__
			void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
											uint   *cellEnd,          // output: cell end index
											float4 *sortedPos,        // output: sorted positions
											uint   *gridParticleHash, // input: sorted grid hashes
											uint   *gridParticleIndex,// input: sorted particle indices
											float4 *oldPos,           // input: sorted position array
											uint    numParticles)
			{
				extern __shared__ uint sharedHash[];    // blockSize + 1 elements
				uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

				uint hash;

				// handle case when no. of particles not multiple of block size
				if (index < numParticles)
				{
					hash = gridParticleHash[index];

					// Load hash data into shared memory so that we can look
					// at neighboring particle's hash value without loading
					// two hash values per thread
					sharedHash[threadIdx.x+1] = hash;

					if (index > 0 && threadIdx.x == 0)
					{
						// first thread in block must load neighbor particle hash
						sharedHash[0] = gridParticleHash[index-1];
					}
				}

				__syncthreads();

				if (index < numParticles)
				{
					// If this particle has a different cell index to the previous
					// particle then it must be the first particle in the cell,
					// so store the index of this particle in the cell.
					// As it isn't the first particle, it must also be the cell end of
					// the previous particle's cell

					if (index == 0 || hash != sharedHash[threadIdx.x])
					{
						cellStart[hash] = index;

						if (index > 0)
							cellEnd[sharedHash[threadIdx.x]] = index;
					}

					if (index == numParticles - 1)
					{
						cellEnd[hash] = index + 1;
					}

					// Now use the sorted index to reorder the pos data
					uint sortedIndex = gridParticleIndex[index];
					float4 pos = oldPos[sortedIndex];
					sortedPos[index] = pos;
				}

			}

			extern "C" __global__
			void setConnectD(   float4  *Pos,
								int     *cons,                  // output: list of particle id's connected to each particle
								float4  *lens,                  // output: length of springs for each connection
								float4  *sortedPos,                // input:  sorted positions
								uint    *gridParticleIndex,     // input:  sorted particle indices
								uint    *cellStart,
								uint    *cellEnd,
								uint     numParticles,
								uint	 numSprings,
								int      gridSizex,
								int      gridSizey,
								int      gridSizez,
								float    worldOriginx,
								float    worldOriginy,
								float    worldOriginz,
								float    voxelSizex,
								float    voxelSizey,
								float    voxelSizez)
			{
				int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
				if (index >= numParticles) return;

				// read particle data from sorted arrays
				float4 pospos = Pos[index];
				float3 pos = make_float3(pospos.x, pospos.y, pospos.z);
				// get address in grid
				int3 gridPos = calcGridPos(pos, make_float3(worldOriginx, worldOriginy, worldOriginz), make_float3(voxelSizex, voxelSizey, voxelSizez));

				uint sub = 0;
				float maxLen = 0.f;
				int maxPos = -1;
				int conSoFar[NUM_SPRINGS];
				float len_limit = sqrtf(voxelSizex*voxelSizex + voxelSizey*voxelSizey + voxelSizez*voxelSizez);

				for (int x=0; x<=(SPRING_SEARCH_SIZE+1)/2; x++)
				for (int mx=-1; mx<=1; mx+=2)
				{
					if (x==0 && mx==1) continue;

					for (int y=0; y<=(SPRING_SEARCH_SIZE+1)/2; y++)
					for (int my=-1; my<=1; my+=2)
					{
						if (y==0 && my==1) continue;

						for (int z=0; z<=(SPRING_SEARCH_SIZE+1)/2; z++)
						for (int mz=-1; mz<=1; mz+=2)
						{
							if (z==0 && mz==1) continue;

							int3 neighbourPos = make_int3(x*mx + gridPos.x, y*my + gridPos.y, z*mz + gridPos.z);
							uint gridHash = calcGridHash(neighbourPos, make_int3(gridSizex, gridSizey, gridSizez));

							// get start of bucket for this cell
							uint startIndex = cellStart[gridHash];
							uint endIndex = cellEnd[gridHash];
							uint countIndex = endIndex - startIndex;

							if (countIndex > 0)
							{
								// iterate over particles in this cell
								for (uint j=startIndex; j<endIndex; j++)
								{
									int con = gridParticleIndex[j];
									bool dynamic = false;
									int tempCon = con;
									float indic = 1.f;
									dynamic = true;

									if ( !dynamic || (index != tempCon) )
									{
										float4 spospos = sortedPos[j];
										float3 pos2 = make_float3(spospos.x - pos.x, spospos.y - pos.y, spospos.z - pos.z);
										float pos2length = sqrtf(pos2.x*pos2.x + pos2.y*pos2.y + pos2.z*pos2.z);
										if (pos2length > len_limit) continue; // skip too long springs	
										if (pos2length < 0.0001f) continue; // skip zero length springs

											for (int c=0; c<sub; c++)
												if (conSoFar[c] == con)
													con = -1;
											
											if (con != -1)
											{
												if (sub < numSprings)
												{
													if (pos2length > maxLen)
													{
														maxLen = pos2length;
														maxPos = sub;
													}
													lens[index*numSprings + sub] = make_float4( pos2.x, pos2.y, pos2.z, indic );
													conSoFar[sub] = con;
													cons[index*numSprings + sub++] = tempCon;
												}
												else if (pos2length < maxLen)
												{
													lens[index*numSprings + maxPos] = make_float4( pos2.x, pos2.y, pos2.z, indic );
													conSoFar[maxPos] = con;
													cons[index*numSprings + maxPos] = tempCon;

													maxLen = 0.f;
													for (int c=0; c<numSprings; c++)
													{
														float3 currentSpring = make_float3( lens[numSprings*index + c].x,
																							lens[numSprings*index + c].y,
																							lens[numSprings*index + c].z );
														float currentSpringLength = sqrtf(currentSpring.x*currentSpring.x + currentSpring.y*currentSpring.y + currentSpring.z*currentSpring.z);
														if (currentSpringLength > maxLen)
														{
															maxLen = currentSpringLength;
															maxPos = c;
														}
													}

												}
											}
									}
								}
							}
						}
					}
				}
			}

			extern "C" __global__
			void findSurfaceD(  float4  *Pos,
								int     *cons,                  // output: list of particle id's connected to each particle
								float4  *lens,                  // output: length of springs for each connection
								uint	*conCount,
								float3  *tempNorms,              // output: temporary normals for each particle
								float	*surface,
								float	*surfVolume,
								uint     numParticles,
								uint	 numSprings,
								int      gridSizex,
								int      gridSizey,
								int      gridSizez,
								float    worldOriginx,
								float    worldOriginy,
								float    worldOriginz,
								float    voxelSizex,
								float    voxelSizey,
								float    voxelSizez)
			{
				int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
				if (index >= numParticles) return;

				float3 kdir = make_float3(0, 0, 0);
				float kcount = 0.0;
				uint springCount = 0;

				for (int s=0; s<numSprings; s++) {

					uint psindex = index*numSprings + s;

					if (cons[psindex] >= 0 && cons[psindex] < numParticles) {
						float4 len = lens[psindex];
						float norm = sqrtf(len.x*len.x + len.y*len.y + len.z*len.z);

						kdir.x += (len.x / norm);
						kdir.y += (len.y / norm);
						kdir.z += (len.z / norm);

						kcount += 1.0;

						springCount += 1;
					}
				}
				conCount[index] = springCount;

				if (kcount > 0) {
					float ksize = sqrtf(kdir.x*kdir.x + kdir.y*kdir.y + kdir.z*kdir.z);
					float3 norm = make_float3(-1. * kdir.x / ksize, -1. * kdir.y / ksize, -1. * kdir.z / ksize);
					ksize = ksize / kcount;

					if (ksize > 0.25) {
						surface[index] = ksize;
						tempNorms[index] = norm;
						float4 pospos = Pos[index];
						float3 pos = make_float3(pospos.x, pospos.y, pospos.z);
						int3 gridPos = calcGridPos(pos, make_float3(worldOriginx, worldOriginy, worldOriginz), make_float3(voxelSizex, voxelSizey, voxelSizez));
						uint gindex = calcGridHash(gridPos, make_int3(gridSizex, gridSizey, gridSizez));
						surfVolume[gindex] = 1;
					} else {
						surface[index] = 0.0;
					}
				}
			}

			extern "C" __global__
			void reduceSpringsD(
								int     *tempCons,                  // output: list of particle id's connected to each particle
								float4  *tempLens,                  // output: length of springs for each connection
								int     *cons,                  // output: list of particle id's connected to each particle
								float4  *lens,                  // output: length of springs for each connection
								uint	*conStart,
								uint	*conCount,
								uint2	*lineSet,
								uint     numParticles,
								uint	 numSprings,
								uint	 totalSprings)
			{
				int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
				if (index >= numParticles) return;

				uint count = conCount[index];
				uint start = conStart[index];

				uint tempindex = index*numSprings;

				for (int s=0; s<count; s++) {
					int con = tempCons[tempindex + s];
					if (con >= 0 && con < numParticles) {
						float4 len = tempLens[tempindex + s];

						cons[start + s] = con;
						lens[start + s] = len;
						lineSet[start + s] = make_uint2(index, con);
					}
				}
			}
			'''
			
		self.makeIsoKernels = '''
			extern "C" __global__ void
			cudaZeroFillerIsotropic( float *include, int d_sizex, int d_sizey, int d_sizez, float ratiox, float ratioy, float ratioz, float threshold, cudaTextureObject_t tex)
			{
				int X = threadIdx.x + blockDim.x*blockIdx.x;
				if (X >= d_sizex) return;
				int Y = threadIdx.y + blockDim.y*blockIdx.y;
				if (Y >= d_sizey) return;
				int Z = blockIdx.z;
				if (Z >= d_sizez) return;

				float targetx, targety, targetz;
				targetx = __int2float_rn(X) * ratiox;
				targety = __int2float_rn(Y) * ratioy;
				targetz = __int2float_rn(Z) * ratioz;
				
				int idx = X + d_sizex * Y + d_sizex * d_sizey * Z;
				
				float spot = tex3D<float>(tex, targetx+0.5f, targety+0.5f, targetz+0.5f);
				
				if (spot > threshold) {
					include[idx] = 1.0;
				}
			}
			'''

	
	
	
	# FUNCTIONS
	def length_float3(self, a):
		return np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
		
	def index2float4(self, index, arraySize, arrayRes):
		page = arraySize[0] * arraySize[1]
		Z = float(np.floor(index / page)) * arrayRes[2]
		Y = float(np.floor((index % page) / arraySize[0])) * arrayRes[1]
		X = float((index % page) % arraySize[0]) * arrayRes[0]
		# vec = float4(X,Y,Z,0.0)
		return [X, Y, Z, 0.0]
		
	def index2float3(self, index, arraySize, arrayRes):
		page = arraySize[0] * arraySize[1]
		Z = float(np.floor(index / page)) * arrayRes[2]
		Y = float(np.floor((index % page) / arraySize[0])) * arrayRes[1]
		X = float((index % page) % arraySize[0]) * arrayRes[0]
		# vec = float4(X,Y,Z,0.0)
		return [X, Y, Z]

	def index2coord(self, index, arraySize):
		page = arraySize[0] * arraySize[1]
		Z = int(index / page)
		Y = int((index % page) / arraySize[0])
		X = int((index % page) % arraySize[0])
		return X,Y,Z
	
	def getParticlePosition(self, i):
		index = self.index_volume[i]
		pos = [self.hPos[4*index+0], self.hPos[4*index+1], self.hPos[4*index+2]]
		return np.array(pos)
	
	def getSurfacePoints(self):
		surfpoint = []
		for i in range(self.numParticles):
			if self.hSurface[i] != 0:
				pos = [self.hPos[4*i+0], self.hPos[4*i+1], self.hPos[4*i+2]]
				surfpoint.append(pos)
		return np.array(surfpoint)

	def getSurfaceNormals(self):
		surfpoint = []
		for i in range(self.numParticles):
			if self.hSurface[i] != 0:
				pos = [self.hNormals[3*i+0], self.hNormals[3*i+1], self.hNormals[3*i+2]]
				surfpoint.append(pos)
		return np.array(surfpoint)
	
	def getVolumePoints(self):
		nppos = self.hPos.reshape((self.numParticles, 4))
		return nppos[:,0:3]		

	def getSDF(self):
		sdf = -2.0 * self.binary_volume + 1
		return sdf.reshape(self.arraySize)

	def getLineSet(self):
		lines = self.hLineSet.reshape((self.totalSprings, 2))	
		somelines = lines[lines[:, 0] < lines[:, 1]]
		return somelines

	def vizLineSet(self):
		line_set = o3d.geometry.LineSet()
		line_set.points = o3d.utility.Vector3dVector(self.getVolumePoints())
		line_set.lines = o3d.utility.Vector2iVector(self.getLineSet())
		colors = [[0, 0.5, 0.5] for i in range(self.totalSprings)]
		line_set.colors = o3d.utility.Vector3dVector(colors)	
		o3d.visualization.draw_geometries([line_set])
	
	def createPCD(self):
		self.pcd = o3d.geometry.PointCloud()
		self.pcd.points = o3d.utility.Vector3dVector(self.getVolumePoints())
	
	def vizVolumePoints(self,make_new=False):
		if not hasattr(self, 'pcd') or make_new:
			self.createPCD()
		o3d.visualization.draw_geometries([self.pcd], point_show_normal=False)

	def createSurfacePCD(self):
		self.surfpcd = o3d.geometry.PointCloud()
		self.surfpcd.points = o3d.utility.Vector3dVector(self.getSurfacePoints())
		self.surfpcd.normals = o3d.utility.Vector3dVector(self.getSurfaceNormals())  # invalidate existing normals

	def vizSurfacePoints(self):
		if not hasattr(self, 'surfpcd'):
			self.createSurfacePCD()	
		o3d.visualization.draw_geometries([self.surfpcd], point_show_normal=True)

	def createTriMesh(self):
		if not hasattr(self, 'surfpcd'):
			self.createSurfacePCD()	
		if not hasattr(self, 'triangles'):
			self.meshifyVolume()
		# self.trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.surfpcd, depth=11, width=0, scale=11, linear_fit=False)[0]
		self.trimesh = o3d.geometry.TriangleMesh()
		self.trimesh.vertices = o3d.utility.Vector3dVector(self.getVolumePoints())
		self.trimesh.triangles = o3d.utility.Vector3iVector(self.triangles)
		self.trimesh.triangle_normals = o3d.utility.Vector3dVector(self.face_normals)
		self.trimesh.paint_uniform_color([0.6, 0.2, 0.6])

	def vizSurfaceTriMesh(self, w_points=False):
		if not hasattr(self, 'trimesh'):
			self.createTriMesh()
		if w_points:
			tripcd = o3d.geometry.PointCloud()
			tripcd.points = o3d.utility.Vector3dVector(self.face_centroids)
			tripcd.normals = o3d.utility.Vector3dVector(np.asarray(self.trimesh.triangle_normals))
			o3d.visualization.draw_geometries([self.trimesh,tripcd], point_show_normal=True, mesh_show_back_face=True,  mesh_show_wireframe=True)
		else:
			o3d.visualization.draw_geometries([self.trimesh], mesh_show_back_face=True,  mesh_show_wireframe=True)

	def getSurfaceTriangles(self):
		if not hasattr(self, 'triangles'):
			self.meshifyVolume()
		return self.triangles

	def createTetraMesh(self):
		if not hasattr(self, 'tetras'):
			self.meshifyVolume()
		self.tetmesh = o3d.geometry.TetraMesh()
		self.tetmesh.vertices = o3d.utility.Vector3dVector(self.getVolumePoints())
		self.tetmesh.tetras = o3d.utility.Vector4iVector(self.tetras)
	
	def vizVolumeTetMesh(self, w_points=False, only_surface_tets=False):
		if not hasattr(self, 'tetras'):
			self.meshifyVolume()
		if not hasattr(self, 'pcd'):
			self.pcd = o3d.geometry.PointCloud()
			self.pcd.points = o3d.utility.Vector3dVector(self.getVolumePoints())
		if only_surface_tets:
			# Find indices of surface particles
			surface_mask = self.hSurface != 0
			surface_indices = set(np.where(surface_mask)[0])
			# Filter tetras: keep if any vertex is a surface particle
			tetras = np.array(self.tetras)
			mask = np.any(np.isin(tetras, list(surface_indices)), axis=1)
			tetras = tetras[mask]
		else:
			tetras = np.array(self.tetras)
		self.tetmesh = o3d.geometry.TetraMesh()
		self.tetmesh.vertices = o3d.utility.Vector3dVector(self.getVolumePoints())
		self.tetmesh.tetras = o3d.utility.Vector4iVector(tetras)
		if w_points:
			if only_surface_tets:
				if not hasattr(self, 'surfpcd'):
					self.createSurfacePCD()	
				o3d.visualization.draw_geometries([self.tetmesh, self.surfpcd])
			else:
				if not hasattr(self, 'pcd'):
					self.createPCD()	
				o3d.visualization.draw_geometries([self.tetmesh, self.pcd])
		else:
			o3d.visualization.draw_geometries([self.tetmesh])

	def findConnections(self):
		m_dCellStart = checkCudaErrors(cudart.cudaMalloc(self.numGridCells * np.dtype(np.uint32).itemsize))
		m_dCellEnd = checkCudaErrors(cudart.cudaMalloc(self.numGridCells * np.dtype(np.uint32).itemsize))

		m_dGridParticleHash = checkCudaErrors(cudart.cudaMalloc(self.numParticles * np.dtype(np.uint32).itemsize))
		m_dGridParticleIndex = checkCudaErrors(cudart.cudaMalloc(self.numParticles * np.dtype(np.uint32).itemsize))
		m_dConCount = checkCudaErrors(cudart.cudaMalloc(self.numParticles * np.dtype(np.uint32).itemsize))

		m_dNZ = checkCudaErrors(cudart.cudaMalloc(self.numGridCells * np.dtype(np.uint32).itemsize))
		checkCudaErrors(cudart.cudaMemcpy(m_dNZ, self.nzindices.astype(np.uint32), self.numParticles * np.dtype(np.uint32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

		m_dVolume = checkCudaErrors(cudart.cudaMalloc(self.numGridCells * np.dtype(np.uint32).itemsize))
		# checkCudaErrors(cudart.cudaMemcpy(m_dVolume, self.binary_volume.astype(np.uint32), self.numGridCells * np.dtype(np.uint32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
		checkCudaErrors(cudart.cudaMemset(m_dVolume, 0, self.numGridCells * np.dtype(np.uint32).itemsize))

		m_dPos = checkCudaErrors(cudart.cudaMalloc(4 * self.numParticles * np.dtype(np.float32).itemsize))
		# checkCudaErrors(cudart.cudaMemcpy(m_dPos, self.hPos.T, 4 * self.numParticles * np.dtype(np.float32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
		m_dSortedPos = checkCudaErrors(cudart.cudaMalloc(4 * self.numParticles * np.dtype(np.float32).itemsize))

		tempCon = np.ones(self.numSprings * self.numParticles, dtype=np.int32) * -1
		m_dTempCon = checkCudaErrors(cudart.cudaMalloc(self.numSprings * self.numParticles * np.dtype(np.int32).itemsize))
		checkCudaErrors(cudart.cudaMemcpy(m_dTempCon, tempCon, self.numSprings * self.numParticles * np.dtype(np.int32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
		m_dTempRests = checkCudaErrors(cudart.cudaMalloc(4 * self.numSprings * self.numParticles * np.dtype(np.float32).itemsize))
		
		setConnectkernelHelper = KernelHelper(self.setConnectKernels, self.devID)
		_cuda_initPosD = setConnectkernelHelper.getFunction(b'initPosD')
		_cuda_calcHashD = setConnectkernelHelper.getFunction(b'calcHashD')
		_cuda_reorderDataAndFindCellStartD = setConnectkernelHelper.getFunction(b'reorderDataAndFindCellStartD')
		_cuda_setConnectD = setConnectkernelHelper.getFunction(b'setConnectD')
		_cuda_findSurfaceD = setConnectkernelHelper.getFunction(b'findSurfaceD')
		_cuda_reduceSpringsD = setConnectkernelHelper.getFunction(b'reduceSpringsD')

		dimBlock = cudart.dim3()
		dimBlock.x = self.deviceProps.maxThreadsPerBlock
		dimBlock.y = 1
		dimBlock.z = 1

		gridx = self.numParticles/dimBlock.x
		if ( self.numParticles % dimBlock.x > 0): gridx += 1
			
		dimGrid = cudart.dim3()
		dimGrid.x = gridx
		dimGrid.y = 1
		dimGrid.z = 1

		kernelArgs = (( m_dPos, m_dNZ, m_dVolume, self.numParticles, 
						self.arraySize[0], self.arraySize[1], self.arraySize[2], 
						self.resolution[0], self.resolution[1], self.resolution[2]), 
						(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, 
						ctypes.c_int, ctypes.c_int, ctypes.c_int, 
						ctypes.c_float, ctypes.c_float, ctypes.c_float,))
		
		checkCudaErrors(cuda.cuLaunchKernel(_cuda_initPosD,
											dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
											dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
											0, 0,                                    # shared mem and stream
											kernelArgs, 0))   
		checkCudaErrors(cudart.cudaDeviceSynchronize())

		checkCudaErrors(cudart.cudaMemcpy(self.index_volume, m_dVolume, self.numGridCells * np.dtype(np.uint32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		print(np.count_nonzero(self.index_volume))
		checkCudaErrors(cudart.cudaFree(m_dVolume))
		checkCudaErrors(cudart.cudaFree(m_dNZ))

		kernelArgs = ((m_dGridParticleHash, m_dGridParticleIndex, m_dPos, self.numParticles, 
						self.arraySize[0], self.arraySize[1], self.arraySize[2], 
						self.worldOrigin[0], self.worldOrigin[1], self.worldOrigin[2], 
						self.resolution[0], self.resolution[1], self.resolution[2]), 
						(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
						ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,))
		
		checkCudaErrors(cuda.cuLaunchKernel(_cuda_calcHashD,
											dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
											dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
											0, 0,                                    # shared mem and stream
											kernelArgs, 0))   
		checkCudaErrors(cudart.cudaDeviceSynchronize())

		checkCudaErrors(cudart.cudaMemset(m_dCellStart, np.iinfo(np.int32).max, self.numGridCells*np.dtype(np.uint32).itemsize))
		checkCudaErrors(cudart.cudaMemset(m_dCellEnd, np.iinfo(np.int32).max, self.numGridCells*np.dtype(np.uint32).itemsize))
		smemSize = np.dtype(np.uint32).itemsize * (dimBlock.x+1)

		kernelArgs = ((m_dCellStart, m_dCellEnd, m_dSortedPos, m_dGridParticleHash, m_dGridParticleIndex, m_dPos, self.numParticles), 
					(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int))
		checkCudaErrors(cuda.cuLaunchKernel(_cuda_reorderDataAndFindCellStartD,
											dimGrid.x, dimGrid.y, dimGrid.z,      	 # grid dim
											dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
											smemSize, 0,                             # shared mem and stream
											kernelArgs, 0))   
		checkCudaErrors(cudart.cudaDeviceSynchronize())

		checkCudaErrors(cudart.cudaMemset(m_dTempCon, np.iinfo(np.int32).max, self.numSprings * self.numParticles * np.dtype(np.int32).itemsize))
		checkCudaErrors(cudart.cudaMemset(m_dTempRests, 0, 4 * self.numSprings * self.numParticles * np.dtype(np.float32).itemsize))

		kernelArgs = ((m_dPos, m_dTempCon, m_dTempRests, m_dSortedPos, m_dGridParticleIndex, 
					   m_dCellStart, m_dCellEnd, self.numParticles, self.numSprings, 
						self.arraySize[0], self.arraySize[1], self.arraySize[2], 
						self.worldOrigin[0], self.worldOrigin[1], self.worldOrigin[2], 
						self.resolution[0], self.resolution[1], self.resolution[2]),
						(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  
	   					 ctypes.c_void_p,  ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
						 ctypes.c_int, ctypes.c_int, ctypes.c_int, 
						 ctypes.c_float, ctypes.c_float, ctypes.c_float, 
						 ctypes.c_float, ctypes.c_float, ctypes.c_float))
		checkCudaErrors(cuda.cuLaunchKernel(_cuda_setConnectD,
											dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
											dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
											0, 0,                                    # shared mem and stream
											kernelArgs, 0))   
		checkCudaErrors(cudart.cudaDeviceSynchronize())

		checkCudaErrors(cudart.cudaFree(m_dCellEnd))
		checkCudaErrors(cudart.cudaFree(m_dGridParticleHash))
		checkCudaErrors(cudart.cudaFree(m_dGridParticleIndex))
		checkCudaErrors(cudart.cudaFree(m_dSortedPos))

		#dCellStart becomes surfVolume
		m_dSurface = checkCudaErrors(cudart.cudaMalloc(self.numParticles * np.dtype(np.float32).itemsize))
		m_dTempNorms = checkCudaErrors(cudart.cudaMalloc(3 * self.numParticles * np.dtype(np.float32).itemsize))
		checkCudaErrors(cudart.cudaMemset(m_dTempNorms, 0, 3 * self.numParticles*np.dtype(np.float32).itemsize))
		checkCudaErrors(cudart.cudaMemset(m_dCellStart, 0, self.numGridCells*np.dtype(np.uint32).itemsize))
		checkCudaErrors(cudart.cudaMemset(m_dSurface, 0, self.numParticles * np.dtype(np.float32).itemsize))

		kernelArgs = ((m_dPos, m_dTempCon, m_dTempRests, m_dConCount, m_dTempNorms,
					   m_dSurface, m_dCellStart, self.numParticles, self.numSprings, 
						self.arraySize[0], self.arraySize[1], self.arraySize[2], 
						self.worldOrigin[0], self.worldOrigin[1], self.worldOrigin[2], 
						self.resolution[0], self.resolution[1], self.resolution[2]),
						(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
	   					 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
						 ctypes.c_int, ctypes.c_int, ctypes.c_int, 
						 ctypes.c_float, ctypes.c_float, ctypes.c_float, 
						 ctypes.c_float, ctypes.c_float, ctypes.c_float))
		checkCudaErrors(cuda.cuLaunchKernel(_cuda_findSurfaceD,
											dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
											dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
											0, 0,                                    # shared mem and stream
											kernelArgs, 0))   
		checkCudaErrors(cudart.cudaDeviceSynchronize())

		checkCudaErrors(cudart.cudaMemcpy(self.hPos, m_dPos, 4 * self.numParticles * np.dtype(np.float32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		checkCudaErrors(cudart.cudaMemcpy(self.hSurface, m_dSurface, self.numParticles * np.dtype(np.float32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		checkCudaErrors(cudart.cudaMemcpy(self.hSurfVolume, m_dCellStart, self.numGridCells * np.dtype(np.uint32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

		self.surfParticles = np.count_nonzero(self.hSurface)
		self.hNormals = np.zeros(self.numParticles * 3, dtype=np.float32)
		checkCudaErrors(cudart.cudaMemcpy(self.hNormals, m_dTempNorms, 3 * self.numParticles * np.dtype(np.float32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		
		checkCudaErrors(cudart.cudaFree(m_dTempNorms))
		checkCudaErrors(cudart.cudaFree(m_dSurface))
		checkCudaErrors(cudart.cudaFree(m_dCellStart))
		checkCudaErrors(cudart.cudaFree(m_dPos))

		checkCudaErrors(cudart.cudaMemcpy(self.hconCount, m_dConCount, self.numParticles * np.dtype(np.uint32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		self.hconStart = np.cumsum(self.hconCount) - self.hconCount
		self.totalSprings = np.sum(self.hconCount).astype(np.int32)

		m_dConStart = checkCudaErrors(cudart.cudaMalloc(self.numParticles * np.dtype(np.uint32).itemsize))
		checkCudaErrors(cudart.cudaMemcpy(m_dConStart, self.hconStart.astype(np.uint32), self.numParticles * np.dtype(np.uint32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
		m_dCon = checkCudaErrors(cudart.cudaMalloc(self.totalSprings * np.dtype(np.int32).itemsize))
		m_dLineSet = checkCudaErrors(cudart.cudaMalloc(self.totalSprings * 2 * np.dtype(np.int32).itemsize))
		m_dRests = checkCudaErrors(cudart.cudaMalloc(4 * self.totalSprings * np.dtype(np.float32).itemsize))
		
		kernelArgs = ((m_dTempCon, m_dTempRests, m_dCon, m_dRests,
					   m_dConStart, m_dConCount, m_dLineSet,
					   self.numParticles, self.numSprings, self.totalSprings), 
						(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	   					 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
						 ctypes.c_int, ctypes.c_int, ctypes.c_int))
		checkCudaErrors(cuda.cuLaunchKernel(_cuda_reduceSpringsD,
											dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
											dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
											0, 0,                                    # shared mem and stream
											kernelArgs, 0))   
		checkCudaErrors(cudart.cudaDeviceSynchronize())

		self.hConnections = np.zeros(self.totalSprings, dtype=np.int32)
		self.hLineSet = np.zeros(2*self.totalSprings, dtype=np.int32)
		# self.hLineSet = self.hLineSet[self.hLineSet[:, 0] < self.hLineSet[:, 1]]
		# self.hLineSet = self.hLineSet[self.hLineSet[:, 0] >= 0]
		# self.hLineSet = self.hLineSet[self.hLineSet[:, 1] < self.numParticles]
		self.hRestLengths = np.zeros(4*self.totalSprings, dtype=np.float32)
		checkCudaErrors(cudart.cudaMemcpy(self.hLineSet, m_dLineSet, self.totalSprings * 2 * np.dtype(np.int32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		checkCudaErrors(cudart.cudaMemcpy(self.hConnections, m_dCon, self.totalSprings * np.dtype(np.int32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		checkCudaErrors(cudart.cudaMemcpy(self.hRestLengths, m_dRests, 4 * self.totalSprings * np.dtype(np.float32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
		
		checkCudaErrors(cudart.cudaFree(m_dConCount))
		checkCudaErrors(cudart.cudaFree(m_dConStart))
		checkCudaErrors(cudart.cudaFree(m_dTempCon))
		checkCudaErrors(cudart.cudaFree(m_dTempRests))
		checkCudaErrors(cudart.cudaFree(m_dCon))
		checkCudaErrors(cudart.cudaFree(m_dRests))

		print(self.totalSprings, " Total Springs")

	def position2coord(self, pos):
		coord = [
			(pos[0] - self.worldOrigin[0]) / self.resolution[0],
			(pos[1] - self.worldOrigin[1]) / self.resolution[1],
			(pos[2] - self.worldOrigin[2]) / self.resolution[2],
		]
		return coord

	def position2index(self, pos):
		coord = self.position2coord(pos)
		index = int( coord[0] + self.arraySize[0] * coord[1] + self.arraySize[0] * self.arraySize[1] * coord[2] )
		return index
	
	def add_face(self, i, j, k):
		key = tuple(sorted((i,j,k)))
		if key not in self.faces:
			self.faces[key] = (i,j,k,len(self.tetras)-1)
		else:
			del self.faces[key]

	def add_triangles(self):
		triangles = np.array(list(self.faces.values()), dtype=int).reshape(-1, 4)
		face_normals = []
		face_coms = []
		for tri in triangles:
			i = tri[0]
			j = tri[1]
			k = tri[2]
			p = self.getParticlePosition(i)
			q = self.getParticlePosition(j)
			r = self.getParticlePosition(k)
			com = (p+q+r)/3.0
			face_coms.append(com)
			qp = (q - p)
			rp = (r - p)
			normal = np.cross(qp,rp)
			magnormal = np.linalg.norm(normal)
			comvec = self.tet_centroids[tri[3]] - com
			cosine_angle = np.dot(normal,comvec) / (magnormal * np.linalg.norm(comvec))
			clipped_cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
			angle_degrees = np.degrees(np.arccos(clipped_cosine_angle))
			if (angle_degrees < 90):
				normal = -1 * normal
			face_normals.append(normal)
		self.triangles = self.index_volume[triangles[:,0:3]]
		self.face_normals = np.array(face_normals)
		self.face_centroids = np.array(face_coms)
		### TODO: 
			
	def tet_insideness(self,i,j,k,l):
		if not self.binary_volume[i]:
			return False
		elif not self.binary_volume[j]:
			return False
		elif not self.binary_volume[k]:
			return False
		elif not self.binary_volume[l]:
			return False
		else:
			return True

	def add_tetra(self, i, j, k, l):
		if not self.tet_insideness(i,j,k,l):
			return
		
		p = self.getParticlePosition(i)
		q = self.getParticlePosition(j)
		r = self.getParticlePosition(k)
		s = self.getParticlePosition(l)
		qp = (q - p) / 10.
		rp = (r - p) / 10.
		sp = (s - p) / 10.
		Dm = np.array((qp,rp,sp)).T
		volume = np.linalg.det(Dm) / 6.0
		if volume < 0.0:
			print('inverted tetrahedral element')
		else:
			self.tetras.append((i,j,k,l))
			self.tet_centroids.append(0.25*(p+q+r+s))
			self.tetvols.append(volume)
			self.totalVolume += volume
			#inv_Dm = np.linalg.inv(Dm)
		 	#self.tet_poses.append(inv_Dm.tolist())
		 	#self.tet_activations.append(0.0)
		 	#self.tet_materials.append((mu,lambda,damping))
			self.add_face(i,k,j)
			self.add_face(j,k,l)
			self.add_face(i,j,l)
			self.add_face(i,l,k)
	
	def grid_index(self, x, y, z):
		return self.arraySize[0] * self.arraySize[1] * z + self.arraySize[0] * y + x 

	def meshifyVolume(self):
		self.faces = {}
		start_vertex = 0

		for z in range(self.arraySize[2]-1):
			for y in range(self.arraySize[1]-1):
				for x in range(self.arraySize[0]-1):
					v0 = self.grid_index(x, y, z) + start_vertex
					v1 = self.grid_index(x + 1, y, z) + start_vertex
					v2 = self.grid_index(x + 1, y, z + 1) + start_vertex
					v3 = self.grid_index(x, y, z + 1) + start_vertex
					v4 = self.grid_index(x, y + 1, z) + start_vertex
					v5 = self.grid_index(x + 1, y + 1, z) + start_vertex
					v6 = self.grid_index(x + 1, y + 1, z + 1) + start_vertex
					v7 = self.grid_index(x, y + 1, z + 1) + start_vertex

					if (x & 1) ^ (y & 1) ^ (z & 1):
						self.add_tetra(v0, v1, v4, v3)
						self.add_tetra(v2, v3, v6, v1)
						self.add_tetra(v5, v4, v1, v6)
						self.add_tetra(v7, v6, v3, v4)
						self.add_tetra(v4, v1, v6, v3)
					else:
						self.add_tetra(v1, v2, v5, v0)
						self.add_tetra(v3, v0, v7, v2)
						self.add_tetra(v4, v7, v0, v5)
						self.add_tetra(v6, v5, v2, v7)
						self.add_tetra(v5, v2, v7, v0)
		
		self.tetras = self.index_volume[self.tetras]
		self.totalVolume = np.round(self.totalVolume, decimals=2)
		self.add_triangles()
		print('*** ', self.tetras.shape[0], " Tetrahedrons")
		print('*** ', self.triangles.shape[0], " Surface Triangles")
		print('*** ', self.totalVolume, " cubic cm Total Volume")

	def trimesh_to_usd(self, name, path):
		if not hasattr(self, 'trimesh'):
			self.createTriMesh()
		roi_usd_path = path + name + '.usda'
		if os.path.isfile(roi_usd_path):
			stage = Usd.Stage.Open(roi_usd_path)
		else:
			stage = Usd.Stage.CreateNew(roi_usd_path)
		root_xform = UsdGeom.Xform.Define(stage, '/Root')
		mesh_path = Sdf.Path(root_xform.GetPath()).AppendChild('primitive')
		mesh = UsdGeom.Mesh.Define(stage, mesh_path)
		mesh.GetPointsAttr().Set(np.asarray(self.trimesh.vertices))
		mesh.CreateFaceVertexCountsAttr(np.full(np.asarray(self.trimesh.triangles).shape[0],3))
		mesh.CreateFaceVertexIndicesAttr(np.asarray(self.trimesh.triangles))
		mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.leftHanded)
		mesh.CreateSubdivisionSchemeAttr().Set("none")
		
		stage.GetRootLayer().Save()

	def tetmesh_to_usd(self, vertices, triangles, name, path, tetras=None):
		roi_usd_path = path + name + '.usda'
		if os.path.isfile(roi_usd_path):
			stage = Usd.Stage.Open(roi_usd_path)
		else:
			stage = Usd.Stage.CreateNew(roi_usd_path)
		root_xform = UsdGeom.Xform.Define(stage, '/Root')
		mesh_path = Sdf.Path(root_xform.GetPath()).AppendChild('primitive')
		mesh = UsdGeom.TetMesh.Define(stage, mesh_path)
		mesh.GetPointsAttr().Set(np.asarray(vertices))
		mesh.CreateTetVertexIndicesAttr().Set(tetras.flatten())
		mesh.CreateSurfaceFaceVertexIndicesAttr().Set(np.asarray(triangles).flatten())
		mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.leftHanded)	
		stage.GetRootLayer().Save()

	def load_usd_mesh(usd_path, mesh_name):
		stage = Usd.Stage.Open(usd_path + mesh_name + '.usda')
		mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/Root/primitive'))
		points = mesh.GetPointsAttr().Get()
		faces = mesh.GetFaceVertexIndicesAttr().Get()
		return points, faces

	def make_isotropic(self, _arraySize, _resolution, _volume, iso_resolution ):
			# send the contour volume, containing the subcontour outlines, to texture memory on the GPU
			# /////////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
			cExtent = cudart.make_cudaExtent(_arraySize[0], _arraySize[1], _arraySize[2])
			channelDesc	= checkCudaErrors(cudart.cudaCreateChannelDesc(32, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat))
			cu_3darray =  checkCudaErrors(cudart.cudaMalloc3DArray(channelDesc, cExtent, cudart.cudaArrayDefault))

			copyParams = cudart.cudaMemcpy3DParms()
			copyParams.srcPos 		= 	cudart.make_cudaPos(0,0,0)
			copyParams.dstPos 		= 	cudart.make_cudaPos(0,0,0)
			copyParams.srcPtr 		=	cudart.make_cudaPitchedPtr(_volume, cExtent.width*np.dtype(np.float32).itemsize, cExtent.width, cExtent.height);
			copyParams.dstArray		= 	cu_3darray
			copyParams.extent       =	cExtent
			copyParams.kind		    =	cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
			checkCudaErrors(cudart.cudaMemcpy3D(copyParams))

			texRes = cudart.cudaResourceDesc()
			texRes.resType = cudart.cudaResourceType.cudaResourceTypeArray
			texRes.res.array.array = cu_3darray

			texCntr = cudart.cudaTextureDesc()
			texCntr.normalizedCoords	=	False
			texCntr.filterMode			=	cudart.cudaTextureFilterMode.cudaFilterModeLinear
			texCntr.addressMode[0]		=	cudart.cudaTextureAddressMode.cudaAddressModeClamp
			texCntr.addressMode[1]		=	cudart.cudaTextureAddressMode.cudaAddressModeClamp
			texCntr.addressMode[2]		=	cudart.cudaTextureAddressMode.cudaAddressModeClamp
			texCntr.readMode = cudart.cudaTextureReadMode.cudaReadModeElementType

			tex = checkCudaErrors(cudart.cudaCreateTextureObject(texRes, texCntr, None))

			iso_resolution = 1 # min(cntr_resolution)
			print(iso_resolution)
			res_ratios = [iso_resolution / _resolution[0], iso_resolution / _resolution[1], iso_resolution / _resolution[2]]
			print(res_ratios)
			iso_arraySize = np.array([int(_arraySize[0]/res_ratios[0] + 0.5), int(_arraySize[1]/res_ratios[1] + 0.5), int(_arraySize[2]/res_ratios[2] + 0.5)])
			print(iso_arraySize)

			dimBlock = cudart.dim3()
			dimBlock.x = 32
			dimBlock.y = 32
			dimBlock.z = 1

			gridx = iso_arraySize[0]/dimBlock.x
			if ( iso_arraySize[0] % dimBlock.x > 0): gridx += 1
			gridy = iso_arraySize[1]/dimBlock.y
			if ( iso_arraySize[1] % dimBlock.y > 0): gridy += 1
			gridz = iso_arraySize[2]
				
			dimGrid = cudart.dim3()
			dimGrid.x = gridx
			dimGrid.y = gridy
			dimGrid.z = gridz

			ivol_size = iso_arraySize[0] * iso_arraySize[1] * iso_arraySize[2] * np.dtype(np.float32).itemsize
			d_vol = checkCudaErrors(cudart.cudaMalloc(ivol_size))
			checkCudaErrors(cudart.cudaMemset(d_vol, 0, ivol_size))

			kernelHelper = KernelHelper(self.makeIsoKernels, self.devID)
			_cudaZeroFillerIsotropic = kernelHelper.getFunction(b'cudaZeroFillerIsotropic')

			kernelArgs = ((d_vol, iso_arraySize[0], iso_arraySize[1], iso_arraySize[2], res_ratios[0], res_ratios[1], res_ratios[2], 10., tex), 
			  (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, None))
			checkCudaErrors(cuda.cuLaunchKernel(_cudaZeroFillerIsotropic,
												dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
												dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
												0, 0,                                    # shared mem and stream
												kernelArgs, 0))   
			checkCudaErrors(cudart.cudaDeviceSynchronize())

			iso_volume = np.zeros(iso_arraySize[0]*iso_arraySize[1]*iso_arraySize[2], dtype=np.float32)
			checkCudaErrors(cudart.cudaMemcpy(iso_volume, d_vol, ivol_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
			
			checkCudaErrors(cudart.cudaDestroyTextureObject(tex))
			checkCudaErrors(cudart.cudaFree(d_vol))
			checkCudaErrors(cudart.cudaFreeArray(cu_3darray))

			return iso_volume