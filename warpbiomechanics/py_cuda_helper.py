# Copyright 2021-2023 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import sys
import numpy as np
import os
from cuda import cuda, cudart, nvrtc


def checkCmdLineFlag(stringRef):
    for i in sys.argv:
        if stringRef == i:
            print('Found ', stringRef)
            return True
    return False


def getCmdLineArgument(stringRef):
    k = 1
    for i in sys.argv:
        if stringRef == i and k < len(sys.argv):
            print(sys.argv[k])
            return sys.argv[k]
        k += 1
    return 0


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(
            result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def findCudaDevice():
    devID = 0
    if checkCmdLineFlag("--device"):
        devID = getCmdLineArgument("--device")
    checkCudaErrors(cudart.cudaSetDevice(devID))
    return devID


def findCudaDeviceDRV():
    devID = 0
    if checkCmdLineFlag("--device"):
        devID = getCmdLineArgument("--device")
    checkCudaErrors(cuda.cuInit(0))
    cuDevice = checkCudaErrors(cuda.cuDeviceGet(devID))
    return cuDevice


class KernelHelper:
    def __init__(self, code, devID):
        prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(
            str.encode(code), b'sourceCode.cu', 0, [], []))
        CUDA_HOME = '/usr/local/cuda'  # os.getenv('CUDA_HOME')
        # print(CUDA_HOME)
        if CUDA_HOME == None:
            CUDA_HOME = os.getenv('CUDA_PATH')
        if CUDA_HOME == None:
            raise RuntimeError(
                'Environment variable CUDA_HOME or CUDA_PATH is not set')
        include_dirs = os.path.join(CUDA_HOME, 'include')

        # Initialize CUDA
        checkCudaErrors(cudart.cudaFree(0))

        major = checkCudaErrors(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID))
        minor = checkCudaErrors(cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID))
        _, nvrtc_minor = checkCudaErrors(nvrtc.nvrtcVersion())
        use_cubin = (nvrtc_minor >= 1)
        prefix = 'sm' if use_cubin else 'compute'
        arch_arg = bytes(
            f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')

        try:
            opts = [b'--fmad=true', arch_arg, '--include-path={}'.format(include_dirs).encode('UTF-8'),
                    b'--std=c++11', b'-default-device']
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b' ' * logSize
            checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode())
            print(err)
            exit(-1)

        if use_cubin:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b' ' * dataSize
            checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
            data = b' ' * dataSize
            checkCudaErrors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = checkCudaErrors(
            cuda.cuModuleLoadData(np.char.array(data)))

    def getFunction(self, name):
        return checkCudaErrors(cuda.cuModuleGetFunction(self.module, name))
