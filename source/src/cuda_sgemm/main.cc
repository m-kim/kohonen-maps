#include <cstdlib>

extern "C" int runCudasGemm(int M, int N);
int M ;
int N ;

int main( int argc, char **argv )
{
	if (argc > 1){
		M = atoi(argv[1]);
		N = atoi(argv[1]);
	}
	else{
		M = 2048;
		N = 2048;
	}


	runCudasGemm(M,N);
};
