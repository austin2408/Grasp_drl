#! /bin/bash

cwd=$PWD
cd $cwd'/sean_approach'

mkdir -p 'datasets'
mkdir -p 'model'

cd 'datasets'
# echo -e "\e[93m download datasets Logger_v3.hdf5 \e[0m"
# gdown --id 1HjoZB_M3njfc9D78QVkAWQpPwNDFTiFS

echo -e "\e[93m download datasets Logger_half_v3_2.hdf5 \e[0m"
gdown --id 1F6HTnuDbP-TCmyxwZVdt0FhnOq-qKe-M

# echo -e "\e[93m download datasets Logger_8.hdf5 \e[0m"
# gdown --id 1UCL-GT43AQ8Ha1Y4NAaWjUD0Di6I4j1P

# echo -e "\e[93m download datasets Logger05_8.hdf5 \e[0m"
# gdown --id 1UrjmRLTVOrWqseT578wWfqdTZbK4Cckw

cd ..
cd 'model'
echo -e "\e[93m download model behavior_half_v3_2_1000.pth \e[0m"
gdown --id 1luTxvmRlhByBrC8q5cnjZoaKwddYF7bM

cd ../..