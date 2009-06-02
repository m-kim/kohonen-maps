#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define DEBUG_PRINT 0

typedef struct{
	float *data;
	int row;
	int col;
} MATRIX;


//float operator[](MATRIX mat, unsigned int index)
//{
//	return mat.data[index];
//}
