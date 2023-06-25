## Update ldd version
https://askubuntu.com/questions/1345342/how-to-install-glibc-2-32-when-i-already-have-glibc2-31
```
mkdir $HOME/glibc/ && cd $HOME/glibc
wget http://ftp.gnu.org/gnu/libc/glibc-2.34.tar.gz
tar -xvzf glibc-2.34.tar.gz
mkdir build 
mkdir glibc-2.34-install
cd build
sudo apt install gawk bison
unset LD_LIBRARY_PATH
~/glibc/glibc-2.34/configure --prefix=$HOME/glibc/glibc-2.34-install
make
make install
```

## Connect
```
./connector -b 158.178.245.66 -d 10.0.0.39 -u ubuntu -k ~/.ssh/ssh-key-2023-06-25.key -s
```
