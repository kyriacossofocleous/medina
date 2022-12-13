
####
#### Testing driver for the f2c script.
####
##########################################


echo "=============================================================="
echo "=================> STEP 1: Copying files. <==================="

(
#set -x
cp raw/* ./messy/smcl/
mkdir -p ./messy/util/medina
cp ../f2c_alpha.py  ./messy/util/medina/.
cp -r ../source  ./messy/util/medina/.
)

echo "=================> STEP 2: Running script. <=================="

(
#set -x
cd messy/util/medina
python ./f2c_alpha.py -r 1 -g 1 -s "../../smcl/" > /dev/null
)

status=$?
if [ $status == 1 ]; then
       echo "Python parser - Unsuccessful"
       exit -1
fi

echo "==========> STEP 3: Compiling the output files. <============="

(
#set -x
cd messy/smcl
nvcc -O0 -DDEBUG -c messy_mecca_kpp_acc.cu  2>&1  | grep error
)

status=$?
if [ $status == 0 ]; then
       echo "NVCC - Unsuccessful"
       exit -1
fi

echo "========> STEP 4: Running CUDA memory check (double) <========"

(
#set -x
cat ./raw/main.c >> ./messy/smcl/messy_mecca_kpp_acc.cu
cd messy/smcl
nvcc -O1 -lineinfo messy_mecca_kpp_acc.cu --keep --keep-dir ./temp_files  2>&1  | grep error
cuda-memcheck ./a.out | grep -v "Results"
)

echo "===> STEP 5: Running the application in double precision. <==="
(
#set -x
cd messy/smcl
nvcc -O1  messy_mecca_kpp_acc.cu --keep --keep-dir ./temp_files  2>&1  | grep error
./a.out | grep -v "Results"
./a.out | grep "Results" | sed -e "s/Results://g" > res_gpu_double.txt
)


status=$?
if [ $status == 1 ]; then
       echo "NVCC - Unsuccessful"
       exit -1
fi


echo "========> STEP 6: Running CUDA memory check (single) <========"

(
#set -x                                                                                                 
mkdir ./messy/smcl_single
mkdir ./messy/smcl_single/temp_files
cp ./messy/smcl/messy_mecca_kpp_acc.cu ./messy/smcl_single/messy_mecca_kpp_acc_single.cu
cd messy/smcl_single
sed -i 's/double/float/g' messy_mecca_kpp_acc_single.cu
sed -i 's/pow/powf/g' messy_mecca_kpp_acc_single.cu
sed -i 's/log10/log10f/g' messy_mecca_kpp_acc_single.cu 
nvcc -O1 -lineinfo messy_mecca_kpp_acc_single.cu --keep --keep-dir ./temp_files  2>&1  | grep error
cuda-memcheck ./a.out | grep -v "Results"
)

echo "===> STEP 7: Running the application in single precision. <==="
(
#set -x                                                                                                 
cd messy/smcl_single
nvcc -O1  messy_mecca_kpp_acc_single.cu --keep --keep-dir ./temp_files  2>&1  | grep error
./a.out | grep -v "Results"
./a.out | grep "Results" | sed -e "s/Results://g" > res_gpu_single.txt
)


status=$?
if [ $status == 1 ]; then
       echo "NVCC - Unsuccessful"
       exit -1
fi




echo "======> STEP 8: Compiling original version in FORTRAN. <======"


(
#set -x
cp raw/*f90 ./messy/fortran
cp raw/main_fortran.c ./messy/fortran
cd messy/fortran
gfortran -c messy_cmn_photol_mem.f90  2>&1  | grep error
gfortran -c messy_main_constants_mem.f90  2>&1  | grep error
gfortran -c messy_mecca_kpp.f90  2>&1  | grep error
gcc      -c main_fortran.c  2>&1  | grep error
gfortran -g *o  -lm
./a.out | grep -v "Results"
./a.out | grep "Results" | sed -e "s/Results://g" > res_fortran.txt
)


echo "==========> STEP 9: Comparing the output results. <==========="

(
#set -x
python compare.py ./messy/fortran/res_fortran.txt messy/smcl/res_gpu_double.txt | grep "Element\|<<<<<<===== WARNING"
python compare.py ./messy/fortran/res_fortran.txt messy/smcl_single/res_gpu_single.txt | grep "Element\|<<<<<<===== WARNING"

)

echo "===========> STEP 10: Cleaning up the directories. <==========="


(
set -x
cd messy/smcl/
rm ./*
rm -r ../smcl_single/
cd ../fortran/
rm ./*
cd ../util/
rm -rf ./*
)

echo "====> Testing Completed"

#EOF
