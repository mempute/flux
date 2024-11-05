
#include "mcpu.h"

intt idx2offset(intt ndim, intt *srank, intt *idx)
{
	intt off = 0, j = 0;

	for(intt i = 1;i < ndim; i++) {//[2,1,1,3]�� ��� ��ũ�� [6,-3.-3, 3]�̰� ��ũ�� 1�� ���� �ΰ� ���� ���ӵ� 3�� �ɼ��� �ߺ��ǰ�
		off += MRANK_SIZE(srank, i) * *(idx + j++);//���ɰ� ������ idx�� [m,0,0,n]�� ���� ��ũ�� 1�� ������ idx�� 0�� ���ۿ� 
	}											//�����Ƿ� ��ũ�� 1�� ������ 0�� ������ off�� �������� 3���� �ߺ��ǰ� �������� �ʴ´�.
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

	for(intt i = 1;i < rdim; i++, j++) {//�ش� �ε����� �ϳ� �Ʒ� ��ũ�� ������� ���� ���̹Ƿ� i�� 1����
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
	sidx[out_axis[j]] = off;//������ �ε����� suf matrix�� ���� �ǰ� ���� �ɼ��� �ȴ�.
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
