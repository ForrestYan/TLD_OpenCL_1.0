run_tld: run_tld.o tld_utils.o TLD.o LKTracker.o FerNNClassifier.o gpu.o
	g++ -o run_tld run_tld.o tld_utils.o TLD.o LKTracker.o FerNNClassifier.o gpu.o `pkg-config opencv --libs` -lOpenCL -L /opt/AMDAPPSDK-3.0/lib/x86 
run_tld.o: run_tld.cpp tld_utils.h TLD.h
	g++ -c run_tld.cpp -o run_tld.o `pkg-config opencv --cflags` -I /opt/AMDAPPSDK-3.0/include 
tld_utils.o: tld_utils.cpp TLD.h
	g++ -c tld_utils.cpp -o tld_utils.o `pkg-config opencv --cflags` -I /opt/AMDAPPSDK-3.0/include 
TLD.o: TLD.cpp TLD.h
	g++ -c TLD.cpp -o TLD.o `pkg-config opencv --cflags` -I /opt/AMDAPPSDK-3.0/include 
LKTracker.o: LKTracker.cpp LKTracker.h
	g++ -c LKTracker.cpp -o LKTracker.o `pkg-config opencv --cflags` -I /opt/AMDAPPSDK-3.0/include 
FerNNClassifier.o: FerNNClassifier.cpp FerNNClassifier.h
	g++ -c FerNNClassifier.cpp -o FerNNClassifier.o `pkg-config opencv --cflags` -I /opt/AMDAPPSDK-3.0/include 
gpu.o: gpu.cpp gpu.h
	g++ -c gpu.cpp -o gpu.o -I /opt/AMDAPPSDK-3.0/include 


clean:
	rm -rf gpu.o FerNNClassifier.o LKTracker.o TLD.o tld_utils.o run_tld.o run_tld
