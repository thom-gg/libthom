- .cu files must not have - in them, use underscores

Build:
```
cmake -B build -DARCH=80
cmake --build build
```

Compile only one routine:
```
cmake -B build -DARCH=80 -DROUTINE=dgemm
cmake --build build
```

Available routines are `reduction`, `sdot`, `sgemv`, `dgemm`, and `hgemm`.
Use `-DROUTINE=all` or omit `ROUTINE` to compile everything.

nvbench requires
CMake > 3.30.4


Build on modal:

wget https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz ;
tar -xzf cmake-3.28.6-linux-x86_64.tar.gz ;
mv cmake-3.28.6-linux-x86_64 /opt/cmake ;
export PATH=/opt/cmake/bin:$PATH ;
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH

cmake -S . -B build -DARCH=80 -DROUTINE=hgemm
cmake --build build


Pour zip son code

tar -cvf routines.tar CMakeLists.txt main.cpp enums.h routines cmake/

# Pour modifier l'image docker
docker build -t dockerthomdocker/cuda-with-cmake:latest .
docker push dockerthomdocker/cuda-with-cmake

# Une fois sur la VM
On host:
scp routines.tar <host>:~

On neocloud instance: 
- tar -xvf routines.tar
- then docker run --gpus all -it --rm -v $(pwd):/workspace  dockerthomdocker/cuda-with-cmake:latest


run_traces() {
  sudo ncu --set full --export report_$2_$1.ncu-rep ./build/libthom test hgemm $2 $1 $1 $1 2 0
}
ncu --set full --export report_cublas_$1.ncu-rep ./build/libthom test hgemm final $1 $1 $1 2 0



sudo nano /etc/modprobe.d/nvidia.conf
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0"  >> /etc/modprobe.d/nvidia.conf




# Cette commande en dessous etati necessaire sinon le reboot suffisait pas
sudo update-initramfs -u ; sudo reboot
