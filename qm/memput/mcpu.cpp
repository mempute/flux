
#include "mcpu.h"

intt idx2offset(intt ndim, intt *srank, intt *idx)
{
	intt off = 0, j = 0;

	for(intt i = 1;i < ndim; i++) {//[2,1,1,3]일 경우 랭크는 [6,-3.-3, 3]이고 랭크가 1인 것이 두개 차원 연속되 3개 옵셋이 중복되거
		off += MRANK_SIZE(srank, i) * *(idx + j++);//계산될것 같으나 idx가 [m,0,0,n]과 같이 랭크가 1인 차원은 idx가 0일 수밖에 
	}											//없으므로 랭크가 1인 차원은 0이 곱해져 off가 위예에서 3개씩 중복되게 증가되지 않는다.
	off += *(idx + j);

	return off;
}
intt sparse_idx2offset(intt ndim, intt *srank, intt *idx, intt *axis)
{
	intt off = 0, j = 0;

	for(intt i = 1;i < ndim; i++) {
		off += MRANK_SIZE(srank, i) * *(idx + *(axis + j++));
	}
	off += *(idx + *(axis + j));

	return off;
}
void offset2idx(intt ndim, intt *srank, intt off, intt *idx)
{
	intt j = 0;

	for(intt i = 1;i < ndim; i++) {
		if(*(srank + j) < 0) *(idx + j++) = 0;
		else {
			*(idx + j++) = off / MRANK_SIZE(srank, i);
			off %= MRANK_SIZE(srank, i);
		}
	}
	*(idx + j) = off;
}
void offset2idx2(intt out_axis[], intt n_preout_axis, intt rdim, intt rrank[], intt off, intt pidx[], intt sidx[])
{
	intt j = 0, k;

	for(intt i = 1;i < rdim; i++, j++) {//해당 인덱스는 하나 아래 랭크의 사이즈로 나눈 몫이므로 i는 1부터
		k = out_axis[j];
		if(j < n_preout_axis) {
			if(rrank[j] < 0) pidx[k] = 0;
			else {
				pidx[k] = off / MRANK_SIZE(rrank, i);
				off %= MRANK_SIZE(rrank, i);
			}
		} else {
			if(rrank[j] < 0) sidx[k] = 0;
			else {
				sidx[k] = off / MRANK_SIZE(rrank, i);
				off %= MRANK_SIZE(rrank, i);
			}
		}
	}
	sidx[out_axis[j]] = off;//마지막 인덱스는 suf matrix의 것이 되고 남은 옵셋이 된다.
}
/*void offset2idx2(intt n_axis, intt axis[], intt n_paxis, intt prank[], intt srank[], intt off, intt pidx[], intt sidx[])
{
	intt j = 0, sz, k;

	for(intt i = 0;i < n_axis; i++) {
		k = axis[i];
		if(i < n_paxis) {
			if(prank[k] < 0) pidx[k] = 0;
			else {
				pidx[k] = off / MRANK_SIZE(prank, k + 1);
				off %= MRANK_SIZE(prank, k);
			}
		} else {
			if(srank[k] < 0) sidx[k] = 0;
			else {
				sidx[k] = off / MRANK_SIZE(srank, k + 1);
				off %= MRANK_SIZE(srank, k);
			}
		}
	}
}*/
void mulf(intt n, intt roff, floatt *m_pari, floatt *m_sari, floatt *m_rari, floatt rplus)
{
	for(;roff < n; roff++) {
		if(rplus == 0) *(m_rari + roff) = 0;
		else *(m_rari + roff) *= rplus;
		*(m_rari + roff) += (*(m_pari + roff) * *(m_sari + roff));
	}
}
