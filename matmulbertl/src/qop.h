#define VDATA_SIZE 16

typedef ap_fixed<16, 0> Dt;

#define Itsize 32
typedef ap_fixed<32, 16> It;

typedef struct v_datatype { Dt data[VDATA_SIZE]; } v_dt;
typedef struct v_inWrrtype{	Dt data[VDATA_SIZE]; } v_arr;
typedef struct v_wbtype{	It data[VDATA_SIZE]; } wb_arr;

enum {
	cores = 1,
	Tp2 = 10,
	Mopers = VDATA_SIZE * 8,
	MopersP2 = 7,
	Tsize = 1<<Tp2,             /* 1024 */
	Nmat = 3,
	Veclen = 14,
};

