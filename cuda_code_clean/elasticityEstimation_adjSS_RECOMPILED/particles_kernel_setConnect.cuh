



// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x % (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y % (params.gridSize.y-1);
    gridPos.z = gridPos.z % (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
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
        float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
        sortedPos[index] = pos;
    }

}




__global__
void setConnectD(   float4  *Pos,
                    uint    *cons,                  // output: list of particle id's connected to each particle
                    float4  *lens,                  // output: length of springs for each connection
                    float4  *sortedPos,                // input:  sorted positions
                    uint    *gridParticleIndex,     // input:  sorted particle indices
                    uint    *cellStart,
                    uint    *cellEnd,
                    uint     dynParticles,
                    uint     statParticles)
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= dynParticles) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(Pos, index));
    // get address in grid
    int3 gridPos = calcGridPos(pos);

    uint sub = 0;
    float maxLen = 0.f;
    uint maxPos = 0xffffffff;
    uint conSoFar[NUM_SPRINGS];

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

                int3 neighbourPos = gridPos + make_int3(x*mx, y*my, z*mz);
                uint gridHash = calcGridHash(neighbourPos);

                // get start of bucket for this cell
                uint startIndex = FETCH(cellStart, gridHash);

                if (startIndex != 0xffffffff)
                {
                    // iterate over particles in this cell
                    uint endIndex = FETCH(cellEnd, gridHash);
                    for (uint j=startIndex; j<endIndex; j++)
                    {
                        uint con = gridParticleIndex[j];
                        //bool dynamic = false;
                        uint tempCon = con;
                        float indic = -1.f;
                        if (con >= statParticles)
                        {
                            //dynamic = true;
                            tempCon = con - statParticles;
                            indic = 1.f;
                        }

                        if ( (index != tempCon) )
                        {
                            float3 pos2 = make_float3(FETCH(sortedPos,j)) - pos;
                            float dist = length(pos2);

                                for (int c=0; c<sub; c++)
                                    if (conSoFar[c] == con)
                                        con = 0xffffffff;
                                if (con != 0xffffffff)
                                {
                                    if (sub < NUM_SPRINGS)
                                    {
                                        if (length(pos2) > maxLen)
                                        {
                                            maxLen = length(pos2);
                                            maxPos = sub;
                                        }
                                        lens[index*NUM_SPRINGS + sub] = make_float4( pos2, indic );
                                        conSoFar[sub] = con;
                                        cons[index*NUM_SPRINGS + sub++] = tempCon;
                                    }
                                    else if (dist < maxLen)
                                    {
                                        lens[index*NUM_SPRINGS + maxPos] = make_float4( pos2, indic );
                                        conSoFar[maxPos] = con;
                                        cons[index*NUM_SPRINGS + maxPos] = tempCon;

                                        maxLen = 0.f;
                                        for (int c=0; c<NUM_SPRINGS; c++)
                                        {
                                            float3 currentSpring = make_float3( lens[NUM_SPRINGS*index + c].x,
                                                                                lens[NUM_SPRINGS*index + c].y,
                                                                                lens[NUM_SPRINGS*index + c].z );
                                            if (length(currentSpring) > maxLen)
                                            {
                                                maxLen = length(currentSpring);
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





