#pragma once

#include "mtrack.h"
#include <iomanip>

#define JUST_SIZEDEF		-2
template<typename DT> class Matrix: public Matrixr {
public:
	DT *mxmHost, *mxmEnd, *mxmDevice, *devGround, *devDock;
	Matrix *ptrLeft, *ptrRight, *mastmat;
	
	void mtxrm(bool r_mut)
	{
		if(mastmat == nullptr) {
			if(mxmHost) free(mxmHost);//cudaFreeHost(mxmHost);
			if(devGround) {
				//printf("cuda groud free %p %p\n", this, devGround);
				CudaDevSet(didground);//�޸� ������ ���� ����̽��� ��Ŀ���Ͽ� �����Ѵ�. �ٷ� �ڿ�
							//������� �����Ǵ� ����̽��� �Ҵ�ǹǷ� �����Ҷ� ��Ŀ�� ����ǵ� �������.
				cudaerror(cudaFree(devGround), "matrix cuda free error");
				cudaDeviceSynchronize();
				devGround = nullptr;
			}
			if(r_mut) {//��� ����
				shadowFree();
				lshadow = nullptr;
				if(devDock) {//��ġ ���� �л� �����Ҷ� �׷����Ʈ ���� �����̺꿡��  
					//printf("cuda dock free %p\n", devDock);
					CudaDevSet(didground);//�����ͷ� sum�Ҷ� �����̺� �׶��尡 �ٸ��� ��������
					cudaerror(cudaFree(devDock), "matrix cuda free error");//�ϱ����� �Ҵ��ߴ� �� ����
					cudaDeviceSynchronize();
					devDock = nullptr;
				}
			}
		}
		if(mxomut && r_mut) {
			mxomut = false;
			CLOSE_MUT_(mutmtx);
		}
	}
	void amxm(intt ndim, intt *axid, sytet init, intt gid = -1, Matrixr *mast = nullptr) //ground �Ҵ翡���� ȣ���
	{
		intt mxsz;
		mxndim = ndim;
		mxshape = axid;
		mastmat = (Matrix *)mast;
		if(init) {
			mlisting();
			mxmHost = nullptr;
			mxmDevice = devGround = devDock = nullptr;
			didground = -1;
			//settleShadow = 0;
			maxmSize = 0;
			lshadow = nullptr;
		} else if(mxmHost) {
			mtxrm(false);
			mxmHost = nullptr;
			mxmDevice = devGround = nullptr;
			didground = -1;
		}
		if(*axid > 0) {
			mxsz = make_rank_sz(ndim, axid, mxranksz);
			if(gid == JUST_SIZEDEF) return;//������ ��� �Ҽ��ְԸ� �ʱ�ȭ�ϰ� ���� �������, ���߿� ��������� �޸� �Ҵ�
			if(mastmat) {//reshape�� ��� reshape�Ǵ� �÷����� �޸𸮸� �����Ѵ�.������� �����ϰ� ó��
				mxmHost = mastmat->mxmHost;
				mxmDevice = mastmat->mxmDevice;
				devGround = mastmat->devGround;
				didground = mastmat->didground;
			} else {
				mxmHost = (DT *)malloc(mxsz * sizeof(DT));
				//cudaError_t error = cudaMallocHost(&mxmHost, mxsz * sizeof(DT));
				//if(error != cudaSuccess) throwFault(-1, "matrix cuda malloc error\n");
				memset(mxmHost, 0x00, mxsz * sizeof(DT));
				//ZeroMemory(mxmHost, mxsz * sizeof(DT));
				if(gid < 0) didground = mxTcr->didTrace;
				else didground = gid;
				if(init) {//�ʱ� ��������� �Ҵ��Ҷ��� ��� ������� ���� �����忡���� �Ҵ�ǰ� ���ڸ��� 
					size_t sz_free = getmfreegpu(didground);//�޸� �Ҵ����� �ʴ´�.
					if(mxsz * sizeof(DT) + mxTcr->DEV_AVAIL_MARGIN >= sz_free) {//�׷����� ���� ����ܰ�
						mxTcr->prompt = 0;//������ ������ üũ�� �Ͽ� �Ҵ��ϹǷ� �������� ���� ����ܰ�
						//printf("xxx %d %d %d\n", sz_free, didground, mxsz);
						return;//���� �Ҵ��� ��츸 ���⼭ üũ�Ǿ� �Ҵ�����̸� ��ý����� �����Ͽ� 
					}			//����ܰ迡�� ���۷��̼��� ������� �ʰ� �ϰ� ���常 �ǰ� �Ѵ�.
				} else {//�׷��� ���� ������� ��� �����ȴ�.�׷��� ���� �����忡�� �Ҵ� �޸�
					CudaDevSet(didground);//����Ͽ� �޸� ���ڸ��� ��� ����.
				}
				//printf("yyy %d %d %p %d %d\n", didground, mxsz, mxTcr, mxTcr->didTrace, gid);
				cudaError_t error = cudaMalloc((void**)&devGround, mxsz * sizeof(DT));
				cudaerror(error, "matrix cuda malloc error");
				cudaMemset(devGround, 0x00, mxsz * sizeof(DT));
				cudaDeviceSynchronize();
				mxmDevice = devGround;
				//groundFocus = true;
				//printf("cuda ground alloc %p %p %d\n", this, devGround, mxsz * sizeof(DT));
			}
			mxmEnd = mxmHost + mxsz;
			maxmSize = mxsz;
			didFocus = didground;
		}
		cpmhot = 0;
	}
	bool resizeing(intt ndim, intt *axid, intt gid)
	{
		intt sz, mxsz;

		SIZE_SHAPE(ndim, axid, sz);
		if(sz > maxmSize) return false;

		mxndim = ndim;
		mxshape = axid;
		mxsz = make_rank_sz(ndim, axid, mxranksz);
		mxmEnd = mxmHost + sz;

		if(gid >= 0 && gid != didground) {
			DT *gptr;
			CudaDevSet(gid);
			cudaError_t error = cudaMalloc((void**)&gptr, mxsz * sizeof(DT));
			cudaerror(error, "matrix cuda malloc error");
			error = cudaMemcpyPeer(gptr, didground, devGround, didground, mxsz * sizeof(DT));
			cudaerror(error, "matrix cuda peer memcpy error");

			CudaDevSet(didground);//���� �޸� ����
			cudaerror(cudaFree(devGround), "matrix cuda free error");
			CudaDevSet(gid);

			mxmDevice = devGround = gptr;
			didFocus = didground = gid;
			cudaDeviceSynchronize();
		}
		return true;
	}
	Matrix(Trace *tcr, ubytet dtp, intt ndim, intt *axid, bool o_mut, intt gid, Matrixr *mast = nullptr)
	{
		mxTcr = tcr;
		mxType = dtp;
		realtied = 0;
		nbackw = 0;
		amxm(ndim, axid, 1, gid, mast);
		mxomut = o_mut;
		mxRutra = rsc::srutra;
		if(o_mut) {
			intt rv;
			CRE_MUT_(mutmtx, rv);
			if(rv < 0) {
				throwFault(-1, (bytex *)"cre mut fail");
			}
		}
	}
	void *devmpoint(intt i)
	{
		switch(i) {
		case 0: return mxmDevice;
		case 1: return devGround;
		case 2: return nullptr;
		case 3: return devDock;
		}
	}
	void devmsetting(intt i, void *devm)
	{
		switch(i) {
		case 0: 
			mxmDevice = (DT *)devm;
			break;
		case 1:
			devGround = (DT *)devm;
			break;
		case 2:
			break;
		case 3:
			devDock = (DT *)devm;
			break;
		}
	}
	void devmcopy(void *tar, void *sor)
	{
		intt sz = MTX_SIZE(this) * sizeof(DT);
		cudaError_t error;
		error = cudaMemcpy((DT *)tar, (DT *)sor, sz, cudaMemcpyDeviceToDevice);
		if(error != cudaSuccess) throwFault(-1, "cuda devm copy error %s\n", cudaGetErrorString(error));
	}
	void groundSet(intt gid)
	{
		if(lshadow) {
			if(gid >= 0 && gid != didground) throwFault(-1, "ground set inconsistant gid\n");
			//groundFocus = true;
			mxmDevice = devGround;
			didFocus = didground;
		}
	}
	void shadowSet(void *dev_shadow, intt id_shadow)
	{
		//groundFocus = false;
		mxmDevice = (DT *)dev_shadow;
		didFocus = id_shadow;
	}
	//���� �ΰ� �Լ��� �÷����� �׶��� �޸𸮿� ������ �Ҵ�ǹǷ� mastmat reshape�� �÷����� �����ϰ� ����
	void dockAlloc(void)
	{
		if(devDock) return;
		intt sz = MTX_SIZE(this) * sizeof(DT);
		CudaDevSet(didground);//�����̺� ���࿡�� ����� �޸𸮴� �׶���� ���� �����ġ�̴�.
								//�����̺� ���� ������� ��� �����ȴ�.
		cudaError_t error = cudaMalloc((void**)&devDock, sz);
		cudaerror(error, "dock alloc error");
		cudaMemset(devDock, 0x00, sz);
		cudaDeviceSynchronize();
	}
	void copyHostToGround(void)
	{
		//����ġ�ϰ�� �ʱ�ɶ� ȣ��Ʈ �޸𸮷� ����ǰ� �ʱ�ȭ ���� �ѹ� ����ġ�� ó�� 
		//������ �ϴ� ������ ȣ��Ʈ->����̽��� ����ȴ�. �׷��� �������� �׷��������忡 ���� ������
		//�޸� ������ ��쿡 ������ �޸𸮿� ȣ��Ʈ�� �ʱ�ȭ ������ ����ǰ� �׶��� �޸𸮴� �ΰ�
		//�����̹Ƿ� �ι�° ������ʹ� �ΰ��� �׶��尡 �����쿡 ����Ǿ� ����ġ ���� ������� �ȴ�.
		//���� �ʱ� ������ �Ҵ����(����ġ ���� �̹� ȣ��Ʈ �޸𸮿� ������)�� �׶��忡 ���� �����Ѵ�.
		CudaDevSet(didground);
		mxmDevice = devGround;
		didFocus = didground;
		CopyHostToDevice(this, 0);
	}
	void arrangeDevice(bool ground_to_shadow, intt did_shadow, void *dev_shadow, bool data)
	{
		intt sz = MTX_SIZE(this) * sizeof(DT);
		cudaError_t error;
		if(ground_to_shadow) {
			//if(data && groundFocus == false) return;
			//groundFocus = false;
			error = cudaMemcpyPeer(dev_shadow, did_shadow, devGround, didground, sz);
			//error = cudaMemcpy(dev_shadow, devGround, sz, cudaMemcpyDeviceToDevice);
			mxmDevice = (DT *)dev_shadow;
			didFocus = did_shadow;
		} else {//shadow to ground
			/*
			printf("iiiiiii\n");
			cpmhot = -1;
			printo(0, 1);
			cout << "\n";

			printf("aaaaaaaaaa %p %p\n", mxmDevice, dev_shadow);
			//cudaMemset(dev_shadow, 0x00, sz);
			mxmDevice = dev_shadow;
			cpmhot = -1;
			printo(0, 1);
			cout << "\n";
			*/
			//if(groundFocus) return;
			//groundFocus = true;
			error = cudaMemcpyPeer(devGround, didground, dev_shadow, did_shadow, sz);
			//error = cudaMemcpy(devGround, dev_shadow, sz, cudaMemcpyDeviceToDevice);
			mxmDevice = devGround;
			didFocus = didground;
			/*
			printf("bbbbbbbbbb\n");
			cpmhot = -1;
			printo(0, 1);
			cout << "\n";
			*/
		}
		if(error != cudaSuccess) throwFault(-1, "arrange device cuda mem copy error %d %s %p\n", ground_to_shadow, cudaGetErrorString(error), dev_shadow);
		cudaDeviceSynchronize();//ffff
	}
	intt sizem(bool t_size = 0)
	{
		return (t_size ? sizeof(DT) : MTX_SIZE(this) * sizeof(DT));
	}
	intt sizem2(intt n, bool bsize = 1) //��ġ n���� �������� ������
	{
		intt sz = MTX_SIZE_RK(this, 1);

		return sz * n * (bsize ? sizeof(DT) : 1);
	}
	void inCopy(Matrixr *i_mtx, sytet gpu) //��Ŀ���� ����̽� �޸𸮴� �׷��� �����忡�� focusDevice���� ����Ǿ���
	{
		Matrix<DT> *imtx = (Matrix<DT> *)i_mtx;
		intt sz = MTX_SIZE(this) * sizeof(DT);

		if(MTX_SIZE(this) != MTX_SIZE(imtx)) throwFault(-1, "size error\n");

		if(realtied) {
			memcpy(mxmHost, imtx->mxmHost, MTX_SIZE(this) * sizeof(DT));
			cudaError_t error = cudaMemcpy(mxmDevice, imtx->mxmHost, sz, cudaMemcpyHostToDevice);
			if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 1 %s\n", cudaGetErrorString(error));
			cpmhot = 0;
		} else {
			if(imtx->cpmhot < 0) {
				if(gpu > 0) {
					cudaError_t error = cudaMemcpy(mxmDevice, imtx->mxmDevice, sz, cudaMemcpyDeviceToDevice);
					if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 2 %s\n", cudaGetErrorString(error));
					cpmhot = -1;
				} else if(gpu < 0 || mxTcr->cpmMode > 0) {
					cudaError_t error = cudaMemcpy(mxmHost, imtx->mxmDevice, sz, cudaMemcpyDeviceToHost);
					if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 3 %s\n", cudaGetErrorString(error));
					cpmhot = 1;
				} else {
					cudaError_t error = cudaMemcpy(mxmDevice, imtx->mxmDevice, sz, cudaMemcpyDeviceToDevice);
					if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 4 %s\n", cudaGetErrorString(error));
					cpmhot = -1;
				}
			} else {
				if(gpu > 0) {
					cudaError_t error = cudaMemcpy(mxmDevice, imtx->mxmHost, sz, cudaMemcpyHostToDevice);
					if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 5 %s\n", cudaGetErrorString(error));
					cpmhot = -1;
				} else if(gpu < 0 || mxTcr->cpmMode >= 0) {
					memcpy(mxmHost, imtx->mxmHost, sz);
					cpmhot = 1;
				} else {
					cudaError_t error = cudaMemcpy(mxmDevice, imtx->mxmHost, sz, cudaMemcpyHostToDevice);
					if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 6 %s\n", cudaGetErrorString(error));
					cpmhot = -1;
				}
			}
		}
	}
	void mmean(intt sz) 
	{
		CopyDeviceToHost2(this, 0, 1, 0);
		*mxmHost /= sz;
		cpmhot = 1;
		CopyHostToDevice2(this, 0, 1, 1);
	}
	intt copyMemory(void *pm, sytet gpu, intt begin, intt size)//��Ŀ���� ����̽� �޸𸮴� �׷��� �����忡�� focusDevice���� ����Ǿ���
	{
		intt sz = (size > 0 ? size : MTX_SIZE(this)) * sizeof(DT);

		if(realtied) {
			memcpy(mxmHost + begin, pm, sz);
			cudaError_t error = cudaMemcpy(mxmDevice + begin, pm, sz, cudaMemcpyHostToDevice);
			if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 7 %s\n", cudaGetErrorString(error));
			cpmhot = 0;
		} else if(gpu > 0) {
			cudaError_t error = cudaMemcpy(mxmDevice + begin, pm, sz, cudaMemcpyHostToDevice);
			if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 8 %s\n", cudaGetErrorString(error));
			cpmhot = -1;
		} else if(gpu < 0 || mxTcr->cpmMode >= 0) {
			memcpy(mxmHost + begin, pm, sz);
			cpmhot = 1;
		} else {
			cudaError_t error = cudaMemcpy(mxmDevice + begin, pm, sz, cudaMemcpyHostToDevice);
			if(error != cudaSuccess) throwFault(-1, "matrix cuda mem copy error 9 %s\n", cudaGetErrorString(error));
			cpmhot = -1;
		}
		return sz;
	}
	void resetMemory(sytet gpu)
	{
		if(maxmSize == 0) return;

		intt sz = MTX_SIZE(this) * sizeof(DT);

		if(realtied) {
			memset(mxmHost, 0x00, sz);
			cudaError_t error = cudaMemset(mxmDevice, 0x00, sz);
			if(error != cudaSuccess) throwFault(-1, "matrix cuda memset error 1 %s %p\n", cudaGetErrorString(error), mxmDevice);
			cpmhot = 0;
		} else if(gpu > 0) {
			cudaError_t error = cudaMemset(mxmDevice, 0x00, sz);
			if(error != cudaSuccess) throwFault(-1, "matrix cuda memset error 2 %s %p\n", cudaGetErrorString(error), mxmDevice);
			cpmhot = -1;
		} else if(gpu < 0 || mxTcr->cpmMode >= 0) {
			memset(mxmHost, 0x00, sz);
			cpmhot = 1;
		} else {
			cudaError_t error = cudaMemset(mxmDevice, 0x00, sz);
			if(error != cudaSuccess) throwFault(-1, "matrix cuda memset error 3 %s %d %p %p\n" , cudaGetErrorString(error), sz, this, mxmDevice);
			cpmhot = -1;
		}
	}
	void *begin_dp(void)
	{
		return mxmDevice;
	}
	void *begin_p(intt off = 0)
	{
		CopyDeviceToHost(this, 0);
		return (mxmHost + off);
	}
	void *begin_wp(intt off = 0)
	{
		CopyDeviceToHost(this, 0);
		cpmhot = 1;//���� ȣ��Ʈ �޸𸮿� ������ ���̹Ƿ�
		return (mxmHost + off);
	}
	void *end_p(void)
	{
		return mxmEnd;
	}
	void *read_p(intt xid[], intt irank, intt *rsz, intt n = 0)
	{
		intt off;

		CopyDeviceToHost(this, 0);
		_idx2offset(mxndim, mxranksz, xid, off);
		if(rsz) {
			*rsz = (irank >= mxndim ? 1 : MRANK_SIZE(mxranksz, irank));
			if(n) {//n�� �־������� irank�� �ϳ� ���� ��ũ�� ����� n�� ���� irank�� n������ ����� ���Ѵ�.
				intt sz = (irank >= mxndim ? 1 : MRANK_SIZE(mxranksz, (irank + 1)));
				if(n * sz < *rsz) *rsz = n * sz;//����� �� ���� ��츸 �缳��, �ִ� irank�� �����ʰ�
			}
		}
		return (mxmHost + off);
	}
	void write_p(intt xid[], void *dat, intt irank, intt wsz = 0)
	{
		intt off;

		CopyDeviceToHost(this, 0);
		_idx2offset(mxndim, mxranksz, xid, off);
		intt sz = (irank >= mxndim ? 1 : MRANK_SIZE(mxranksz, irank));//������ ��ũ�� ������
		memcpy(mxmHost + off, dat, (wsz ? (wsz > sz ? sz : wsz) : sz) * sizeof(DT));//wsz - ������ ��ũ�� ����� ��ü ������ ������
		CopyHostToDevice(this, 1);
	}
	void mrandn(TContext *tcxt, doublet m, doublet a, intt n)
	{
		/*if(n < 10000001) {
			random_device rd;
			mt19937 mt(rd());
			if(this->mxType == memput::mp::tfloat) {
				normal_distribution<floatt> initd1(m, a);
				for(intt i = 0; i < n; i++) {
					*(mxmHost + i) = (floatt)initd1(mt);
				}
			} else if(this->mxType == memput::mp::tdouble) {
				normal_distribution<doublet> initd1(m, a);
				for(intt i = 0; i < n; i++) {
					*(mxmHost + i) = (doublet)initd1(mt);
				}
			} else throwFault(-1, "rand type error\n");
			CopyHostToDevice(this, 1);
			return;
		}*/
		OneVar onev;
		*(doublet *)&onev.idxOne[0] = m;
		*(doublet *)&onev.idxOne[2] = a;
		mone(tcxt, nullx, this, 0, &onev, AOP_RANDOM, RAND_T_N, PDC3, 0);//gpu������ ����� ũ�� �������Ƿ� cpu�� �����ϰ�
	}
	void mrandu(TContext *tcxt, doublet m, doublet a, intt n)
	{
		/*if(n < 10000001) {
			random_device rd;
			mt19937 mt(rd());
			if(this->mxType == memput::mp::tint) {
				uniform_int_distribution<intt> initd1(m, a);
				for(intt i = 0; i < n; i++) {
					*(mxmHost + i) = (intt)initd1(mt);
				}
			} else if(this->mxType == memput::mp::tfloat) {
				uniform_real_distribution<floatt> initd1(m, a);
				for(intt i = 0; i < n; i++) {
					*(mxmHost + i) = (floatt)initd1(mt);
				}
			} else if(this->mxType == memput::mp::tdouble) {
				uniform_real_distribution<doublet> initd1(m, a);
				for(intt i = 0; i < n; i++) {
					*(mxmHost + i) = (doublet)initd1(mt);
				}
			} else throwFault(-1, "rand type error\n");
			CopyHostToDevice(this, 1);
			return;
		}*/
		OneVar onev;
		*(doublet *)&onev.idxOne[0] = m;
		*(doublet *)&onev.idxOne[2] = a;
		mone(tcxt, nullx, this, 0, &onev, AOP_RANDOM, RAND_T_U, PDC3, 0);//gpu������ ����� ũ�� �������Ƿ� cpu�� �����ϰ�
	}
	intt uniform(TContext *tcxt, Univ *uni)
	{//���� �Լ����� ��� �ʱⰪ �����ϴ� ���̹Ƿ� ���� cpu/gpu�޸� ��� �ʴ´�.
		intt len = MTX_SIZE(this);

		switch(uni->opvuni) {
		case FILL_UNIV_OP:
			r_copy_val_type(mxmHost, &uni->cvuni, mxType, len);
			break;
		case ARANGE_UNIV_OP:
			if(uni->cvuni > 0 && len != uni->cvuni) throwFault(-1, "matrix size inconsistant\n");
			{
				DT a = 0;//a = 1;//ABP_BWTEST
				for(intt i = 0;i < len; i++) *(mxmHost + i) = a++;
			}
			break;
		case RANDN_UNIV_OP:
			mrandn(tcxt, *(doublet *)&uni->cvuni, *(doublet *)&uni->cvuni2, len);
			return len;
		case RANDU_UNIV_OP:
			mrandu(tcxt, *(doublet *)&uni->cvuni, *(doublet *)&uni->cvuni2, len);
			return len;
		case DSTR_WRITE_OP:
		{
			bytet *dstr = (bytet *)uni->cvuni, digit[24];
			intt i, j, dlen = strlen(dstr), k;

			for(i = j = 0;j < len; j++) {
				for(;i < dlen && (*(dstr + i) < '0' || *(dstr + i) > '9') && *(dstr + i) != '.' && *(dstr + i) != '-'; i++);
				if(i == dlen) break;
				for(k = 0;i < dlen && (('0' <= *(dstr + i) && *(dstr + i) <= '9') || *(dstr + i) == '.' || *(dstr + i) == '-'); i++) {
					digit[k++] = *(dstr + i);
				}
				digit[k] = '\0';
				*(mxmHost + j) = (DT)atof(digit);
				//for(;*(dstr + i) == ' ' || *(dstr + i) == '\t' || *(dstr + i) == '\n' || *(dstr + i) == '\r'; i++);
			}
		}
			break;
		case TYPED_WRITE_OP:
		{
			bytet *src = (bytet *)uni->cvuni;
			ubytet tsrc = uni->tpvuni;
			r_adj_val_type(mxmHost, src, mxType, tsrc, len);
		}
			break;
		case COPY_H2D_OP:
			break;
		case EXPONENT_UNIV_OP:
		{
			intt exp_c = (intt)uni->cvuni;
			DT a = 1;
			for(intt i = 0;i < len; i++, a *= exp_c) *(mxmHost + i) = a;
		}
			break;
		case EXPAND_UNIV_OP:
		{
			Matrix<DT> *src = (Matrix<DT> *)uni->cvuni;
			intt n = *(intt *)&uni->cvuni2, axis = *((intt *)&uni->cvuni2 +1);
			intt off = 0;

			CopyDeviceToHost(src, 0);//ȣ��Ʈ �޸𸮿��� ���� �о�� �ϹǷ� ����ȭ
			if(src->mxndim <= axis) {//�� ���� ���� Ȯ��
				intt sn = MTX_SIZE(src);
				DT a;
				for(intt i = 0;i < sn; i++) {//�ҽ��� ���� �� ������
					a = *(src->mxmHost + i);
					for(intt j = 0;j < n; j++) *(mxmHost + off++) = a;//Ÿ�ٿ� n���� �ߺ��Ͽ� �����Ѵ�.
				}
			} else {
				intt sz = MTX_SIZE_RK(src, axis);
				intt sn = MTX_SIZE(src) / sz;//�ҽ��� ������ ������� ��ü ����� ������ �ֻ��� ����� ����
				DT *p;
				for(intt i = 0;i < sn; i++) {
					p = src->mxmHost + i * sz;//�ҽ��� �� �������� ������ ����������� 
					for(intt j = 0;j < n; j++) {
						memcpy(mxmHost + off, p, sz * sizeof(DT));//Ÿ�ٿ� n���� �ߺ��Ͽ� �����Ѵ�.
						off += sz;
					}
				}
			}
		}
			break; 
		case MIN_MAX_V_OP:
		{
			DT minv, maxv;

			CopyDeviceToHost(this, 0);//ȣ��Ʈ �޸𸮿��� ���� �о�� �ϹǷ� ����ȭ
			minv = maxv = *mxmHost;
			for(intt i = 0; i < len; i++) {
				if(*(mxmHost + i) < minv) minv = *(mxmHost + i);
				else if(*(mxmHost + i) > maxv) maxv = *(mxmHost + i);
			}
			*(doublet *)&uni->cvuni = (doublet)minv;
			*(doublet *)&uni->cvuni2 = (doublet)maxv;
			return len;
		}
			break;
		case STD_NORMAL:
		{
			DT mean = 0, d, stv = 0, std;

			CopyDeviceToHost(this, 0);//ȣ��Ʈ �޸𸮿��� ���� �о�� �ϹǷ� ����ȭ
			for(intt i = 0; i < len; i++) mean += *(mxmHost + i);//�Է°� �հ�
			mean /= len;
			for(intt i = 0; i < len; i++) {
				d = (*(mxmHost + i) - mean);//��� ����
				stv += d * d;//�л�, ������� ���� ��
			}
			std = 1.0 / std::sqrt(stv / len + 1e-9);//ǥ������(�л����) ����
			for(intt i = 0; i < len; i++) {//�Է°� ��� ������ ǥ������ ������ ���Ͽ� ǥ�ذ� ����
				*(mxmHost + i) = (*(mxmHost + i) - mean) * std;
			}
			if(uni->tpvuni == 2 || uni->tpvuni == 3) CopyHostToDevice(this, 1);
			return len;
		}
			break;
		case X_NORMAL:
		{
			bool reverse = *(sytet *)&uni->cvuni, avg_pre = *((sytet *)&uni->cvuni + 1);
			Matrix<DT> *origin = (Matrix<DT> *)uni->cvuni2;
			intt nfeat = SZ_MTX_LOW_FIRST(this), nseq, szseq;
			DT curv, gv, *prev = (DT *)malloc(nfeat * sizeof(DT)), *psum = nullx;
			
			if(avg_pre || origin) {//���� �������� ���ĺ� ����� prev������ �̹� �������� ���ĺ� ���ϰ�� ����
				if(mxndim < 2) throwFault(-1, "dims small\n");
				szseq = SZ_MTX_LOW_SECOND(this);
				nseq = szseq / nfeat;
				if(avg_pre) {
					psum = (DT *)malloc(nfeat * sizeof(DT));
					for(intt i = 0; i < nfeat; i++) *(psum + i) = 0;
				}
			}
			for(intt i = 0; i < nfeat; i++) *(prev + i) = 0.5;//�ʱ� prev������ 0.5

			CopyDeviceToHost(this, 0);//ȣ��Ʈ �޸𸮿��� ���� �о�� �ϹǷ� ����ȭ
			if(origin) CopyDeviceToHost(origin, 0);
			for(intt i = 0; i < len; i++) {
				if(reverse) {//���ϰ�������κ��� ������ ����
					gv = *(mxmHost + i);
					//������ 0�� ������ ��ü �̵��Ͽ� ����� �����. (prev�� 0���̸� ���ϰ����
					//1�̵Ǿ� ������ �ȵȴ�, ���� �����͸� ����� ��ȯ)
					curv = ((*(prev + i % nfeat) + 1e-9) * gv) / ((1 - gv) + 1e-9);
					*(mxmHost + i) = curv;
					//printf("%d:%d %f %f %f\n", i / nfeat, i % nfeat, *(prev + i % nfeat), gv, curv);
				} else {//�����ͷκ��� ���ϰ�� ���� ����
					curv = *(mxmHost + i);
					*(mxmHost + i) = curv / (*(prev + i % nfeat) + curv);//���ϰ�� ����
					//printf("%d:%d %f %f %f %f\n", i / nfeat, i % nfeat, *(prev + i % nfeat), curv, *(mxmHost + i), *(prev + i % nfeat) + curv);
				}
				if(avg_pre) {//������ ������ ������ ����
					if(origin) curv = *(origin->mxmHost + i);//���������̰� �������� �־�������
					*(psum + i % nfeat) += curv;//�������� ������ ����	//���������� ������ ����
					intt j = i + 1;
					if(j >= szseq) {//ù��° �������� �ʃ��� ���������� �ϹǷ� �ι�° ���������� ������ ����.
						if((j / nfeat) % nseq == 0) {//������������ ������ ������ ������ ���Ŀ���
							//�����Ͽ� ���� �������� ù��° ������ ���� ���ĺ� ���� ������ ��հ���
							*(prev + j % nfeat) = *(psum + j % nfeat) / nseq;//����Ͽ�
							*(psum + j % nfeat) = 0;						//������ ����
						}
					} else *(prev + i % nfeat) = curv;//ù��° �������� ���ĺ��� ���� �����۰��� ���������� ����.
				} else if(origin) {//�����۴��� ������ �����̰� ���������̰� �������� �־�������
					intt j = i + nfeat;//���� �������� ������ ������ ���� ���ĺ��� ��������
					if(j >= szseq && (j / nfeat) % nseq == 0) {//������ ������ ���� ����������
						//����, �� ������������ ù��° �����۸� �������� ���� �������� ������ �����۰���
						*(prev + i % nfeat) = *(origin->mxmHost + i);//���������� ����.
					} else *(prev + i % nfeat) = curv;//�� �̿ܿ� �̹����� ������ ���������� ����
				} else {//���ĺ��� �̹� �������� ���� ���� �������� ���������� ����.(���������̸�
					*(prev + i % nfeat) = curv;//�̹� �������̰� ��� ���Ⱚ�̸� �̹� �����Ͱ�)
				}
			}
			free(prev);
			if(avg_pre) free(psum);
			if(uni->tpvuni == 2 || uni->tpvuni == 3) CopyHostToDevice(this, 1);
			return len;
		}
			break;
		case SIN_POSITIONAL:
		{
			intt k = 0, nseq = uni->cvuni ? uni->cvuni : DIM_MTX_LOW_SECOND(this), ndim = DIM_MTX_LOW_FIRST(this);
			
			for(intt i = 0; i < nseq; i++) {
				for(intt j = 0; j < ndim; j++) {
					*(mxmHost + k) = (k % 2 == 0 ? 
						std::sin((floatt)i / std::pow(10000, 2 * (floatt)j / ndim)) :
						std::cos((floatt)i / std::pow(10000, 2 * (floatt)j / ndim)) );
					k++;
					//printf("%f ", 2 * (floatt)j / ndim);
					//printf("%f ", std::pow(10000, 2 * j / ndim));
					//printf("%f ", i / std::pow(10000, 2 * j / ndim));
				}
				//printf("\n");
			}
			break;
		}
		default:
			throwFault(-1, "uniform not command\n");
		}
		CopyHostToDevice(this, 1);
		return len;
	}
	void printr(ostream &output, intt rank, intt off, sytet width) {
		intt n = *(mxshape + rank);
		output << "[";
		if(n < mxTcr->npAbrib) {
			for(int j = 0; j < n; j++)  output << setw(width) << std::left << *(mxmHost + off + j) << " ";
		} else {
			for(int j = 0; j < 3; j++)  output << setw(width) << std::left << *(mxmHost + off + j) << " ";
			cout << "..., ";
			for(int j = n - 2; j < n; j++)  output << setw(width) << std::left << *(mxmHost + off + j) << " ";
		}
		output << "]";
	}
	void matp_sib(ostream &output, intt rank, intt &off, bool &newline, sytet leaf_one, sytet width)
	{
		intt n = *(mxshape + rank);
		intt i, j;

		if(n < mxTcr->npAbrib || leaf_one == 2) {
			for(i = 0;i < n; i++) {
				matp_depth(output, rank + 1, off, i == n - 1 ? true : false, newline, leaf_one, width);
				if(i < n - 1) {
					if(leaf_one && *(mxshape + mxndim - 1) == 1) j = 2;//�ǳ�(����) ������ 1�̸� �������� �ʰ� ������ ����
					else j = 1;
					for(;j < mxndim - rank; j++) {
						output << "\n";
						newline = true;
					}
				}
			}
		} else {
			for(i = 0; i < 5; i++) {
				matp_depth(output, rank + 1, off, false, newline, leaf_one, width);
				if(leaf_one && *(mxshape + mxndim - 1) == 1) j = 2;//�ǳ�(����) ������ 1�̸� �������� �ʰ� ������ ����
				else j = 1;
				for(;j < mxndim - rank; j++) {
					output << "\n";
					newline = true;
				}
			}
			output << "     ...," << endl;
			if(j != 2) output << "\n" << endl;
			for(;i < n - 5; i++) off += MTX_SIZE_RK(this, rank +1);
			for(; i < n; i++) {
				matp_depth(output, rank + 1, off, i == n - 1 ? true : false, newline, leaf_one, width);
				if(i < n - 1) {
					if(leaf_one && *(mxshape + mxndim - 1) == 1) j = 2;//�ǳ�(����) ������ 1�̸� �������� �ʰ� ������ ����
					else j = 1;
					for(;j < mxndim - rank; j++) {
						output << "\n";
						newline = true;
					}
				}
			}
		}
	}
	void matp_depth(ostream &output, intt rank, intt &off, bool last, bool &newline, sytet leaf_one, sytet width)
	{
		if(newline) {
			for(intt j = 0;j < rank; j++) output << " ";
			newline = false;
		}
		if(rank + 1 == mxndim) {
			printr(output, rank, off, width);
			off += MRANK_SIZE(mxranksz, rank);
		} else {
			output << "[";
			matp_sib(output, rank, off, newline, leaf_one, width);
			output << "]";
		}
	}
	void printo(sytet leaf_one = 0, sytet width = 1)
	{
		intt off = 0;
		bool newline = false;
		CopyDeviceToHost(this, 0);
		matp_depth(cout, 0, off, true, newline, leaf_one, width);
	}
	void iprinto(intt i)
	{
		CopyDeviceToHost(this, 0);
		cout << "[";
		cout << *(mxmHost + i);
		cout << "]";
	}
	void debugStamp(const bytex *msg)
	{
		if(msg) printf("$[%s]$\n", msg);
	}
	void mdot(TContext *tcxt, Matrixr *sdot_mtx, Matrixr *rdot_mtx, DotVar *pdotv, sytet trans_order, bool dot1, void *rplus)
	{
		Matrix<DT> *smtx = (Matrix<DT> *)sdot_mtx, *rmtx = (Matrix<DT> *)rdot_mtx;
		SignalR *sr = rsc::srutra->srGet();
		DotTrack<DT> *dtrk;
		intt i = 0, n, width;
		DotVar *dotv;
		unit lap = (lapType == 2 ? xucurrenttime() : 0);
		bool gpu = mxRutra->policyTrack(MTX_SIZE(rmtx), 0, 0, pdotv->shareUnit, width, n, mxTcr->cpmMode);
		if(pdotv->bwGetOri && rmtx->nbackw <= 1) rplus = nullx;
		SyncHstDev(gpu, rplus, this, smtx, rmtx);
		if(pdotv->useCublas && n == 1 && gpu) {
			dtrk = (DotTrack<DT> *)rsc::srutra->trkGet<DotTrack<DT>>(tcxt, 0, -1, 2);
			dtrk->mSetDtra(this, smtx, rmtx, -1, rplus ? ((DT *)rplus == (DT *)1 ? 1 : *(DT *)rplus) : 0);
			dtrk->ordTrans = pdotv->transOrder;
			dtrk->dtGstr = pdotv;
			dtrk->ontrack(sr, CUBLAS_TRACTH);
		} else {//�� ���̽��� trans order�ʿ����.(���� �����Ͽ� ���� �ϴ� ����� �� �������̹Ƿ�) 
			tcxt->cxbegin();	
			dotv = (DotVar *)tcxt->cxalloc(sizeof(DotVar), i);
			memcpy(dotv, pdotv, sizeof(DotVar));
			//dotv->szShrinkSuf = MTX_SIZE(smtx) / dotv->nJointAxis;//dot version 1
			tcxt->syncCxt2Dev(gpu);
			for(i = 0;i < n; i++) {
				dtrk = (DotTrack<DT> *)rsc::srutra->trkGet<DotTrack<DT>>(tcxt, i, width, gpu);
				dtrk->mSetDtra(this, smtx, rmtx, dot1, rplus ? ((DT *)rplus == (DT *)1 ? 1 : *(DT *)rplus) : 0);
				dtrk->ontrack(sr, gpu);
			}
		}
		sr->srWait();
		if(sr->afterCopy == 1) {
			CopyDeviceToHost(rmtx, 1);
		} else if(sr->afterCopy == 2) CopyHostToDevice(rmtx, 1);
		if(lapType == 2) printf("dot[%d] lap: %lld\n", MTX_SIZE(rmtx), xucurrenttime() - lap);
		sr->srPut<DotTrack<DT>>();
	}
	void marith(TContext *tcxt, Matrixr *sar_mtx, Matrixr *rar_mtx, ArithVar *parv, ubytet tval, void *sval, void *rplus, sytet aop)
	{
		Matrix<DT> *smtx = (Matrix<DT> *)sar_mtx, *rmtx = (Matrix<DT> *)rar_mtx;
		SignalR *sr = rsc::srutra->srGet();
		ArithTrack<DT> *atrk;
		intt i = 0, n, width;
		ArithVar *arv;
		unit lap = (lapType == 2 ? xucurrenttime() : 0);
		bool gpu = mxRutra->policyTrack(MTX_SIZE(rmtx), 0, 0, PDN, width, n, mxTcr->cpmMode);

		msetArithv(smtx, rmtx, parv, tval);
		tcxt->cxbegin();
		arv = (ArithVar *)tcxt->cxalloc(sizeof(ArithVar), i);
		if(parv == nullx) {//������ ���� ���ο��� �ܼ� ��� ���� ȣ��
			arv->paintVar = 0;
			msetArithv(smtx, rmtx, arv, tval);
		} else memcpy(arv, parv, sizeof(ArithVar));

		tcxt->syncCxt2Dev(gpu);
		if(arv->bwGetOri && rmtx->nbackw <= 1) rplus = nullx;
		SyncHstDev(gpu, rplus, this, (smtx && this != smtx ? smtx : ((Matrix *)0)), rmtx);
		for(i = 0;i < n; i++) {
			atrk = (ArithTrack<DT> *)rsc::srutra->trkGet<ArithTrack<DT>>(tcxt, i, width, gpu);
			atrk->mSetAtra(this, smtx, rmtx, tval, sval ? *(DT *)sval : 0, aop, rplus ? ((DT *)rplus == (DT *)1 ? 1 : *(DT *)rplus) : 0);
			atrk->ontrack(sr, gpu);
		}
		sr->srWait();
		if(lapType == 2) printf("arith[%d] lap: %lld\n", MTX_SIZE(rmtx), xucurrenttime() - lap);
		sr->srPut<ArithTrack<DT>>();
	}
	void msplit(TContext *tcxt, void *psplit_mtx, intt nsplit, intt axis, bool stacking, bool bw, bool parity)
	{
		Matrix<DT> *mtx = nullx;
		SignalR *sr = rsc::srutra->srGet();
		ConcatTrack<DT> *ctrk;
		ConcatVar *ccv;
		void *dev_mptr, *host_mptr;
		intt i, *prank_size, n, width;
		unit lap = (lapType == 2 ? xucurrenttime() : 0);
		bool gpu = mxRutra->policyTrack(MTX_SIZE(this), 0, 0, PDC, width, n, mxTcr->cpmMode);
		bool each_map;

		tcxt->cxbegin();
		ccv = (ConcatVar *)tcxt->cxalloc(sizeof(ConcatVar), i);
		prank_size = tcxt->cxalloc(mxndim * sizeof(intt), ccv->szRankPrimary);
		memcpy(prank_size, mxranksz, mxndim * sizeof(intt));
		tcxt->cxalign();
		dev_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrDevSecondary);
		host_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrHostSecondary);
		bool b = 0;
		if(parity) {
			for(i = 0;i < nsplit; i++) {
				mtx = *((Matrix<DT> **)psplit_mtx + i);
				if(mtx->nbackw > 1) b = 1;//�Ѱ��� ����� ��Ƽ���� ��� ��ġ��Ų��.
				*((DT **)dev_mptr + i) = mtx->mxmDevice;
				*((DT **)host_mptr + i) = mtx->mxmHost;
				//printf("%p\n", mtx->mxmDevice);
			}
		} else if(gpu && mxshape[axis] < 30) {//���� ���� ������� 30�� ���ϾƸ� �������� �� ���Ҹ��� �ε����� ������ �ٷ� ã���Ѵ�.
			intt inner_sz = (axis == mxndim - 1 ? 1 : MRANK_SIZE(mxranksz, axis + 1));
			intt k = 0;
			dev_mptr = tcxt->cxalloc(mxshape[axis] * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(mxshape[axis] * sizeof(void *), ccv->mptrHostSecondary);
			intt *sdim = tcxt->cxalloc(mxshape[axis] * sizeof(intt), ccv->sdimCat);
			intt *sbase = tcxt->cxalloc(mxshape[axis] * sizeof(intt), ccv->sbaseCat);
			intt sbaseoff = 0;
			for(i = 0;i < nsplit; i++) {
				mtx = *((Matrix<DT> **)psplit_mtx + i);
				if(mtx->nbackw > 1) b = 1;//�Ѱ��� ����� ��Ƽ���� ��� ��ġ��Ų��.
				for(intt j = 0;j < mtx->mxshape[axis]; j++, k++) {
					*((DT **)dev_mptr + k) = mtx->mxmDevice;
					*((DT **)host_mptr + k) = mtx->mxmHost;
					*((intt *)sdim + k) = mtx->mxshape[axis] * inner_sz;
					*((intt *)sbase + k) = sbaseoff;
				}
				sbaseoff += mtx->mxshape[axis] * inner_sz;
			}
			each_map = 1;
		} else {//���� ���� ������� 30�� ����ũ�� ���� �������� �ε����� �����Ѵ�.
			intt inner_sz = (axis == mxndim - 1 ? 1 : MRANK_SIZE(mxranksz, axis + 1));
			dev_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrHostSecondary);
			intt *sdim = tcxt->cxalloc(nsplit * sizeof(intt), ccv->sdimCat);
			intt *sbase = tcxt->cxalloc(nsplit * sizeof(intt), ccv->sbaseCat);
			intt sbaseoff = 0;
			for(i = 0;i < nsplit; i++) {
				mtx = *((Matrix<DT> **)psplit_mtx + i);
				if(mtx->nbackw > 1) b = 1;//�Ѱ��� ����� ��Ƽ���� ��� ��ġ��Ų��.
				*((DT **)dev_mptr + i) = mtx->mxmDevice;
				*((DT **)host_mptr + i) = mtx->mxmHost;
				*((intt *)sdim + i) = mtx->mxshape[axis] * inner_sz;
				*((intt *)sbase + i) = sbaseoff;
				//printf("%p %d %d\n", mtx->mxmHost, mtx->mxshape[axis] * inner_sz, sbaseoff);
				sbaseoff += mtx->mxshape[axis] * inner_sz;
			}
			each_map = 0;
		}
		if(b == 0) bw = 0;//��� �n���� ��Ƽ ������ �ƴϸ� ����� �÷��׸� ���½�Ų��.split���꿡�� Ÿ�� �������� ���⸸�ǹǷ�
		tcxt->syncCxt2Dev(gpu);

		if(gpu) {
			CopyHostToDevice(this, 0);
			if(bw) {//����� �����̰� �Ѱ��� �n���� ��Ƽ �����̸� ��ä split��Ʈ������ cpu,gpu �޸𸮸� ��ġ��Ų��.
				for(i = 0;i < nsplit; i++) {
					mtx = *((Matrix<DT> **)psplit_mtx + i);
					CopyHostToDevice(mtx, 0);
				}
			}
		} else {
			CopyDeviceToHost(this, 0);
			if(bw) {//����� �����̰� �Ѱ��� �n���� ��Ƽ �����̸� ��ä split��Ʈ������ cpu,gpu �޸𸮸� ��ġ��Ų��.
				for(i = 0;i < nsplit; i++) {
					mtx = *((Matrix<DT> **)psplit_mtx + i);
					CopyDeviceToHost(mtx, 0);
				}
			}
		}
		if(parity == 0) {
			if(each_map) nsplit = mxshape[axis] * -1;
			else nsplit *= -1;
		}
		for(i = 0;i < n; i++) {
			ctrk = (ConcatTrack<DT> *)rsc::srutra->trkGet<ConcatTrack<DT>>(tcxt, i, width, gpu);
			ctrk->mSetCatra(this, nsplit, parity ? *(mxshape + axis) / nsplit : each_map, axis, mtx->mxndim, false, psplit_mtx, bw);
			ctrk->ontrack(sr, gpu);
		}
		sr->srWait();
		if(sr->afterCopy == 1) {
			for(i = 0;i < nsplit; i++) {
				mtx = *((Matrix<DT> **)psplit_mtx + i);
				CopyDeviceToHost(mtx, 1);
			}
		} else if(sr->afterCopy == 2) {
			for(i = 0;i < nsplit; i++) {
				mtx = *((Matrix<DT> **)psplit_mtx + i);
				CopyHostToDevice(mtx, 1);
			}
		}
		if(lapType == 2) printf("split[%d] lap: %lld\n", MTX_SIZE(this), xucurrenttime() - lap);
		sr->srPut<ConcatTrack<DT>>();
	}
	void mconcat(TContext *tcxt, void *pcat_mtx, intt ncat, intt axis, bool stacking, bool bw, bool parity)
	{
		Matrix<DT> *mtx = nullx;
		SignalR *sr = rsc::srutra->srGet();
		intt i, *prank_size, n, width;
		ConcatTrack<DT> *ctrk;
		ConcatVar *ccv;
		void *dev_mptr, *host_mptr;
		unit lap = (lapType == 2 ? xucurrenttime() : 0);
		bool gpu = mxRutra->policyTrack(MTX_SIZE(this), 0, 0, PDC, width, n, mxTcr->cpmMode);
		bool each_map;

		tcxt->cxbegin();
		ccv = (ConcatVar *)tcxt->cxalloc(sizeof(ConcatVar), i);
		prank_size = tcxt->cxalloc(mxndim * sizeof(intt), ccv->szRankPrimary);
		memcpy(prank_size, mxranksz, mxndim * sizeof(intt));
		tcxt->cxalign();
		if(parity) {//���� ����� �յ��� ���� ��ħ
			dev_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrHostSecondary);
			for(i = 0;i < ncat; i++) {
				mtx = *((Matrix<DT> **)pcat_mtx + i);
				*((DT **)dev_mptr + i) = mtx->mxmDevice;
				*((DT **)host_mptr + i) = mtx->mxmHost;
			}
		} else if(gpu && mxshape[axis] < 30) {//���� ���� ������� 30�� ���ϾƸ� �������� �� ���Ҹ��� �ε����� ������ �ٷ� ã���Ѵ�.
			intt inner_sz = (axis == mxndim - 1 ? 1 : MRANK_SIZE(mxranksz, axis + 1));
			intt k = 0;
			dev_mptr = tcxt->cxalloc(mxshape[axis] * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(mxshape[axis] * sizeof(void *), ccv->mptrHostSecondary);
			intt *sdim = tcxt->cxalloc(mxshape[axis] * sizeof(intt), ccv->sdimCat);
			intt *sbase = tcxt->cxalloc(mxshape[axis] * sizeof(intt), ccv->sbaseCat);
			intt sbaseoff = 0;
			for(i = 0;i < ncat; i++) {
				mtx = *((Matrix<DT> **)pcat_mtx + i);
				for(intt j = 0;j < mtx->mxshape[axis]; j++, k++) {
					*((DT **)dev_mptr + k) = mtx->mxmDevice;
					*((DT **)host_mptr + k) = mtx->mxmHost;
					*((intt *)sdim + k) = mtx->mxshape[axis] * inner_sz;
					*((intt *)sbase + k) = sbaseoff;
				}
				sbaseoff += mtx->mxshape[axis] * inner_sz;
			}
			each_map = 1;
		} else {//���� ���� ������� 30�� ����ũ�� ���� �÷��� �������� �ε����� �����Ѵ�.
			intt inner_sz = (axis == mxndim - 1 ? 1 : MRANK_SIZE(mxranksz, axis + 1));
			dev_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrHostSecondary);
			intt *sdim = tcxt->cxalloc(ncat * sizeof(intt), ccv->sdimCat);
			intt *sbase = tcxt->cxalloc(ncat * sizeof(intt), ccv->sbaseCat);
			intt sbaseoff = 0;
			for(i = 0;i < ncat; i++) {
				mtx = *((Matrix<DT> **)pcat_mtx + i);
				*((DT **)dev_mptr + i) = mtx->mxmDevice;
				*((DT **)host_mptr + i) = mtx->mxmHost;
				*((intt *)sdim + i) = mtx->mxshape[axis] * inner_sz;
				*((intt *)sbase + i) = sbaseoff;
				//printf("%p %d %d\n", mtx->mxmHost, mtx->mxshape[axis] * inner_sz, sbaseoff);
				sbaseoff += mtx->mxshape[axis] * inner_sz;
			}
			each_map = 0;
		}
		tcxt->syncCxt2Dev(gpu);
		if(nbackw <= 1) bw = 0;//Ÿ���� �n���� ��Ƽ ������ �ƴϸ� ����� �÷��׸� ���½�Ų��.concat���꿡�� Ÿ�� �������� ���⸸�ǹǷ�
		if(gpu) {
			if(bw) CopyHostToDevice(this, 0);//target
			for(i = 0;i < ncat; i++) {
				mtx = *((Matrix<DT> **)pcat_mtx + i);//source
				CopyHostToDevice(mtx, 0);
			}
		} else {
			if(bw) CopyDeviceToHost(this, 0);//target
			for(i = 0;i < ncat; i++) {
				mtx = *((Matrix<DT> **)pcat_mtx + i);//source
				CopyDeviceToHost(mtx, 0);
			}
		}
		if(parity == 0) {
			if(each_map) ncat = mxshape[axis] * -1;
			else ncat *= -1;
		}
		for(i = 0;i < n; i++) {
			ctrk = (ConcatTrack<DT> *)rsc::srutra->trkGet<ConcatTrack<DT>>(tcxt, i, width, gpu);
			ctrk->mSetCatra(this, ncat, parity ? *(mxshape + axis) / ncat : each_map, axis, mtx->mxndim, true, nullx, bw);
			ctrk->ontrack(sr, gpu);
		}
		sr->srWait();
		if(lapType == 2) printf("concat[%d] lap: %lld\n", MTX_SIZE(this), xucurrenttime() - lap);
		sr->srPut<ConcatTrack<DT>>();
	}
	void mtranspose(TContext *tcxt, void *ret_mtx, TransVar *ptsvar, bool bw)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)ret_mtx;
		SignalR *sr = rsc::srutra->srGet();
		TransposeTrack<DT> *trs;
		TransVar *tsvar;
		intt i, n, width;
		bool gpu = mxRutra->policyTrack(MTX_SIZE(rmtx), 0, 0, PDC, width, n, mxTcr->cpmMode);

		tcxt->cxbegin();
		tsvar = (TransVar *)tcxt->cxalloc(sizeof(TransVar), i);
		memcpy(tsvar, ptsvar, sizeof(TransVar));
		tcxt->syncCxt2Dev(gpu);
		if(rmtx->nbackw <= 1) bw = 0;//Ÿ���� �n���� ��Ƽ ������ �ƴϸ� ����� �÷��׸� ���½�Ų��.
		SyncHstDev(gpu, bw, this, ((Matrix *)0), rmtx);
		for(i = 0;i < n; i++) { 
			trs = (TransposeTrack<DT> *)rsc::srutra->trkGet<TransposeTrack<DT>>(tcxt, i, width, gpu);
			trs->mSetTrans(this, rmtx, bw);
			trs->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<TransposeTrack<DT>>();
	}
	void msoftmax(TContext *tcxt, Matrixr *rmtx, Matrixr *sum_mtx, Matrixr *max_mtx, Matrixr *buf_mtx, OprVar1 *poprv)
	{
		SignalR *sr = rsc::srutra->srGet();
		SoftmaxTrack<DT> *trc;
		OprVar1 *oprv;
		intt i, n, width;
		intt feat_sz = SZ_MTX_LOW_FIRST(rmtx);
		bool gpu = mxRutra->policyTrack(MTX_SIZE(rmtx), 1, feat_sz, PDG, width, n, mxTcr->cpmMode);

		if(poprv) {
			tcxt->cxbegin();
			oprv = (OprVar1 *)tcxt->cxalloc(sizeof(OprVar1), i);
			memcpy(oprv, poprv, sizeof(OprVar1));
			tcxt->syncCxt2Dev(gpu);
		}
		sum_mtx->resetMemory(gpu ? 1 : -1);
		max_mtx->resetMemory(gpu ? 1 : -1);
		SyncHstDev(gpu, 1, this, ((Matrix *)sum_mtx), ((Matrix *)max_mtx));//max_mtx�� ���ϸ�Ʈ������ �ƴ� �б������Ǵ� ���̹Ƿ� rcopy 1�� ����
		for(i = 0;i < n; i++) {
			trc = (SoftmaxTrack<DT> *)rsc::srutra->trkGet<SoftmaxTrack<DT>>(tcxt, i, width, gpu);
			trc->mSetSoftx(this, (Matrix<DT> *)rmtx, (Matrix<DT> *)sum_mtx, (Matrix<DT> *)max_mtx, (Matrix<DT> *)buf_mtx, feat_sz);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<SoftmaxTrack<DT>>();
	}
	void msoftx_cross_e(TContext *tcxt, void *ret_mtx, void *tar_mtx)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)ret_mtx;
		SignalR *sr = rsc::srutra->srGet();
		SoftmaxCrossETrack<DT> *trc;
		intt i, n, width;
		intt feat_sz = SZ_MTX_LOW_FIRST(this);
		bool gpu = mxRutra->policyTrack(MTX_SIZE(this), 0, feat_sz, PDG, width, n, mxTcr->cpmMode);//rmtx�� this���� feat_sz�� ���� ��ҵ� ��Ʈ����
		
		rmtx->resetMemory(gpu ? 1 : -1);
		SyncHstDev(gpu, 1, this, ((Matrix *)tar_mtx), rmtx);//�ٷ������� ������ rmtx�� �ݿ��ϱ����� rcopy 1�� ����.
		for(i = 0;i < n; i++) {
			trc = (SoftmaxCrossETrack<DT> *)rsc::srutra->trkGet<SoftmaxCrossETrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, rmtx, tar_mtx, feat_sz);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<SoftmaxCrossETrack<DT>>();
	}
	void mmean_square_e(TContext *tcxt, Matrixr *tar_mtx, void *ret_mtx, bool mean)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)ret_mtx;
		Matrix<DT> *tmtx = (Matrix<DT> *)tar_mtx;
		SignalR *sr = rsc::srutra->srGet();
		MeanSquareETrack<DT> *trc;
		intx i, n, width;
		bool gpu;
		if(mean) {
			gpu = mxRutra->policyTrack3(MTX_SIZE(this), width, n, mean, mxTcr->cpmMode);//��ü �� ����� ���Ѵ� ���̹Ƿ� cpu�� ����ó�� ���ϰ�
			rmtx->resetMemory(gpu ? 1 : -1);
		} else gpu = mxRutra->policyTrack(MTX_SIZE(this), -1, nullx, PDG, width, n, mxTcr->cpmMode);
		SyncHstDev(gpu, 1, this, tmtx, rmtx);//�ٷ������� ������ rmtx�� �ݿ��ϱ����� rcopy 1�� ����.
		for(i = 0;i < n; i++) {
			trc = (MeanSquareETrack<DT> *)rsc::srutra->trkGet<MeanSquareETrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, tmtx, rmtx, mean);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<MeanSquareETrack<DT>>();
	}
	void msum(TContext *tcxt, void *ret_mtx, void *cmul, bool mean)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)ret_mtx;
		SignalR *sr = rsc::srutra->srGet();
		SumTrack<DT> *trc;
		intx i, n, width;
		bool gpu = mxRutra->policyTrack3(MTX_SIZE(this), width, n, mean, mxTcr->cpmMode);
		rmtx->resetMemory(gpu ? 1 : -1);
		SyncHstDev(gpu, 1, this, ((Matrix *)0), rmtx);//�ٷ������� ������ rmtx�� �ݿ��ϱ����� rcopy 1�� ����.
		for(i = 0;i < n; i++) {
			trc = (SumTrack<DT> *)rsc::srutra->trkGet<SumTrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, ret_mtx, cmul, mean);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<SumTrack<DT>>();
	}
	void moptadm(TContext *tcxt, void *v_mtx, void *g_mtx, void *r_mtx, floatt beta1, 
		floatt beta2, floatt lr, floatt e, intt dec)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)r_mtx;
		SignalR *sr = rsc::srutra->srGet();
		OptAdmTrack<DT> *trc;
		intt i, n, width;
		bool gpu = mxRutra->policyTrack(MTX_SIZE(rmtx), 0, 0, PDN, width, n, mxTcr->cpmMode);
		if(gpu) {
			CopyHostToDevice(this, 0);
			CopyHostToDevice(((Matrix *)v_mtx), 0);
			CopyHostToDevice(((Matrix *)g_mtx), 0);
			CopyHostToDevice(rmtx, 0);
		} else {
			CopyDeviceToHost(this, 0);
			CopyDeviceToHost(((Matrix *)v_mtx), 0);
			CopyDeviceToHost(((Matrix *)g_mtx), 0);
			CopyDeviceToHost(rmtx, 0);
		}
		for(i = 0;i < n; i++) {
			trc = (OptAdmTrack<DT> *)rsc::srutra->trkGet<OptAdmTrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, v_mtx, g_mtx, rmtx, beta1, beta2, lr, e, dec);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<OptAdmTrack<DT>>();
	}
	void moptsgd(TContext *tcxt, void *r_mtx, floatt lr, intt dec)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)r_mtx;
		SignalR *sr = rsc::srutra->srGet();
		OptSgdTrack<DT> *trc;
		intt i, n, width;
		bool gpu = mxRutra->policyTrack(MTX_SIZE(rmtx), 0, 0, PDN, width, n, mxTcr->cpmMode);
		SyncHstDev(gpu, 1, this, ((Matrix *)0), rmtx);//����ɱ� rmtx�� �б� �����ǹǷ� rcopy 1
		for(i = 0;i < n; i++) {
			trc = (OptSgdTrack<DT> *)rsc::srutra->trkGet<OptSgdTrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, rmtx, lr, dec);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<OptSgdTrack<DT>>();
	}
	void mone(TContext *tcxt, Matrixr *s_mtx, Matrixr *r_mtx, sytet isz, OneVar *pvar, intt aop, intt aop2, sytet pdiv, sytet rplus, intt feat_sz = 0)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)r_mtx;
		Matrix<DT> *smtx = (Matrix<DT> *)s_mtx;
		SignalR *sr = rsc::srutra->srGet();
		OneTrack<DT> *trc;
		intt i, n, width, sz;
		unit lap = (lapType == 2 ? xucurrenttime() : 0);
		if(isz == 0) sz = MTX_SIZE(this);
		else if(isz == 1) sz = MTX_SIZE(smtx);
		else sz = MTX_SIZE(rmtx);
		//���������� ���� �ݵ�� gpu�� ó���ؾ� �ϴ� ���̽��̸�(PDC3) pdiv�� 1�� ����.
		bool gpu = mxRutra->policyTrack(sz, 0, feat_sz, pdiv, width, n, pdiv == PDC3 ? 1 : mxTcr->cpmMode);
		if(pvar) {
			void *p;
			tcxt->cxbegin();
			p = tcxt->cxalloc(sizeof(OneVar), i);
			memcpy(p, pvar, sizeof(OneVar));
			tcxt->syncCxt2Dev(gpu);
		}
		if(rplus > 1) rmtx->resetMemory(gpu ? 1 : -1);
		else if(rplus > 0 && rmtx->nbackw <= 1) rplus = 0;//rplus�� �����̸� ������ ��ġ
		SyncHstDev(gpu, rplus, this, smtx, rmtx);
		for(i = 0;i < n; i++) {
			trc = (OneTrack<DT> *)rsc::srutra->trkGet<OneTrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, smtx, rmtx, aop, aop2, sz, rplus);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		if(sr->afterCopy == 1) {
			CopyDeviceToHost(rmtx, 1);
		} else if(sr->afterCopy == 2) CopyHostToDevice(rmtx, 1);
		if(lapType == 2) printf("mone[%d] lap: %lld\n", sz, xucurrenttime() - lap);
		sr->srPut<OneTrack<DT>>();
	}
	void mtwo(TContext *tcxt, Matrixr *s_mtx, Matrixr *r_mtx, Matrixr *bp_mtx, Matrixr *bs_mtx, intt aop, intt aop2)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)r_mtx;
		Matrix<DT> *smtx = (Matrix<DT> *)s_mtx;
		Matrix<DT> *bpmtx = (Matrix<DT> *)bp_mtx;
		Matrix<DT> *bsmtx = (Matrix<DT> *)bs_mtx;
		SignalR *sr = rsc::srutra->srGet();
		TwoTrack<DT> *trc;
		intt i, n, width;
		unit lap = (lapType == 2 ? xucurrenttime() : 0);
		bool gpu = mxRutra->policyTrack(MTX_SIZE(this), 0, 0, PDC, width, n, mxTcr->cpmMode);
		
		SyncHstDev(gpu, 0, this, smtx, rmtx);
		if(bp_mtx) SyncHstDev(gpu, 0, bpmtx, bsmtx, rmtx);
		for(i = 0;i < n; i++) {
			trc = (TwoTrack<DT> *)rsc::srutra->trkGet<TwoTrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, smtx, rmtx, bpmtx, bsmtx, aop, aop2);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		sr->srPut<TwoTrack<DT>>();
	}
	void mclip(TContext *tcxt, Matrixr *r_mtx, doublet low, doublet high)
	{
		OneVar onev;

		*(doublet *)&onev.idxOne[0] = low;
		*(doublet *)&onev.idxOne[2] = high;

		mone(tcxt, nullx, r_mtx, 2, &onev, AOP_TYPE1, TYPE1_CLIP, PDC, 0);
	}
	void mlayer_norm(TContext *tcxt, Matrixr *r_mtx, Matrixr *md, Matrixr *mz, Matrixr *mv, Matrixr *mean,
		 Matrixr *g_mz, Matrixr *var, Matrixr *mdmean, Matrixr *ga, Matrixr *be, Matrixr *g_ga, Matrixr *g_be, bool bw)
	{
		Matrix<DT> *rmtx = (Matrix<DT> *)r_mtx;
		SignalR *sr = rsc::srutra->srGet();
		OneTrack<DT> *trc;
		intt i, n, width;
		intt rsz = MTX_SIZE(rmtx);
		intt feat_sz = SZ_MTX_LOW_FIRST(rmtx);
		bool gpu = mxRutra->policyTrack(rsz, 0, feat_sz, PDN, width, n, mxTcr->cpmMode);

		((Matrix<DT> *)mean)->resetMemory(gpu ? 1 : -1);
		if(bw) {
			((Matrix<DT> *)var)->resetMemory(gpu ? 1 : -1);
			((Matrix<DT> *)mdmean)->resetMemory(gpu ? 1 : -1);
		} else {
			((Matrix<DT> *)mv)->resetMemory(gpu ? 1 : -1);
		}
		//ga, be, g_ga, g_be�� ����ġ�̹Ƿ� run�������� �ʱ�ȭ ��.
		tcxt->cxbegin();
		intt **p = (intt **)tcxt->cxalloc(sizeof(intt *) * 11, i);
		i = 0;
		if(gpu) {
			CopyHostToDevice(this, 0);
			CopyHostToDevice(((Matrix<DT> *)mean), 0);
			CopyHostToDevice(((Matrix<DT> *)ga), 0);
			CopyHostToDevice(((Matrix<DT> *)mv), 0);
			*(p + i++) = (intt *)((Matrix<DT> *)md)->mxmDevice;
			*(p + i++) = (intt *)((Matrix<DT> *)mz)->mxmDevice;
			*(p + i++) = (intt *)((Matrix<DT> *)mv)->mxmDevice;
			*(p + i++) = (intt *)((Matrix<DT> *)mean)->mxmDevice;
			if(bw) {
				CopyHostToDevice(((Matrix<DT> *)md), 0);
				CopyHostToDevice(((Matrix<DT> *)mz), 0);
				CopyHostToDevice(((Matrix<DT> *)var), 0);
				CopyHostToDevice(((Matrix<DT> *)mdmean), 0);
				CopyHostToDevice(((Matrix<DT> *)g_ga), 0);
				CopyHostToDevice(((Matrix<DT> *)g_be), 0);
				*(p + i++) = (intt *)((Matrix<DT> *)g_mz)->mxmDevice;
				*(p + i++) = (intt *)((Matrix<DT> *)var)->mxmDevice;
				*(p + i++) = (intt *)((Matrix<DT> *)mdmean)->mxmDevice;
				*(p + i++) = (intt *)((Matrix<DT> *)ga)->mxmDevice;
				*(p + i++) = nullx;
				*(p + i++) = (intt *)((Matrix<DT> *)g_ga)->mxmDevice;
				*(p + i++) = (intt *)((Matrix<DT> *)g_be)->mxmDevice;
			} else {
				CopyHostToDevice(((Matrix<DT> *)be), 0);
				*(p + i++) = nullx;
				*(p + i++) = nullx;
				*(p + i++) = nullx;
				*(p + i++) = (intt *)((Matrix<DT> *)ga)->mxmDevice;
				*(p + i++) = (intt *)((Matrix<DT> *)be)->mxmDevice;
				*(p + i++) = nullx;
				*(p + i++) = nullx;
			}
		} else {
			CopyDeviceToHost(this, 0);
			CopyDeviceToHost(((Matrix<DT> *)mean), 0);
			CopyDeviceToHost(((Matrix<DT> *)ga), 0);
			CopyDeviceToHost(((Matrix<DT> *)mv), 0);
			*(p + i++) = (intt *)((Matrix<DT> *)md)->mxmHost;
			*(p + i++) = (intt *)((Matrix<DT> *)mz)->mxmHost;
			*(p + i++) = (intt *)((Matrix<DT> *)mv)->mxmHost;
			*(p + i++) = (intt *)((Matrix<DT> *)mean)->mxmHost;
			if(bw) {
				CopyDeviceToHost(((Matrix<DT> *)md), 0);
				CopyDeviceToHost(((Matrix<DT> *)mz), 0);
				CopyDeviceToHost(((Matrix<DT> *)var), 0);
				CopyDeviceToHost(((Matrix<DT> *)mdmean), 0);
				CopyDeviceToHost(((Matrix<DT> *)g_ga), 0);
				CopyDeviceToHost(((Matrix<DT> *)g_be), 0);
				*(p + i++) = (intt *)((Matrix<DT> *)g_mz)->mxmHost;
				*(p + i++) = (intt *)((Matrix<DT> *)var)->mxmHost;
				*(p + i++) = (intt *)((Matrix<DT> *)mdmean)->mxmHost;
				*(p + i++) = (intt *)((Matrix<DT> *)ga)->mxmHost;
				*(p + i++) = nullx;
				*(p + i++) = (intt *)((Matrix<DT> *)g_ga)->mxmHost;
				*(p + i++) = (intt *)((Matrix<DT> *)g_be)->mxmHost;
			} else {
				CopyDeviceToHost(((Matrix<DT> *)be), 0);
				*(p + i++) = nullx;
				*(p + i++) = nullx;
				*(p + i++) = nullx;
				*(p + i++) = (intt *)((Matrix<DT> *)ga)->mxmHost;
				*(p + i++) = (intt *)((Matrix<DT> *)be)->mxmHost;
				*(p + i++) = nullx;
				*(p + i++) = nullx;
			}
		}
		for(i = 0;i < n; i++) {
			trc = (OneTrack<DT> *)rsc::srutra->trkGet<OneTrack<DT>>(tcxt, i, width, gpu);
			trc->mSetTrack(this, nullx, rmtx, AOP_LAYNOR, feat_sz, rsz, bw);
			trc->ontrack(sr, gpu);
		}
		sr->srWait();
		if(sr->afterCopy == 1) {
			if(bw) {
				CopyDeviceToHost(((Matrix<DT> *)g_ga), 1);
				CopyDeviceToHost(((Matrix<DT> *)g_be), 1);
			} else {
				CopyDeviceToHost(((Matrix<DT> *)md), 1);
				CopyDeviceToHost(((Matrix<DT> *)mv), 1);
				CopyDeviceToHost(((Matrix<DT> *)mz), 1);
			}
			CopyDeviceToHost(rmtx, 1);
		} else if(sr->afterCopy == 2) {
			if(bw) {
				CopyHostToDevice(((Matrix<DT> *)g_ga), 1);
				CopyHostToDevice(((Matrix<DT> *)g_be), 1);
			} else {
				CopyHostToDevice(((Matrix<DT> *)md), 1);
				CopyHostToDevice(((Matrix<DT> *)mv), 1);
				CopyHostToDevice(((Matrix<DT> *)mz), 1);
			}
			CopyHostToDevice(rmtx, 1);
		}
		sr->srPut<OneTrack<DT>>();
	}
	void mdiag_mul(TContext *tcxt, Matrixr *s_mtx, Matrixr *r_mtx)
	{
		OneVar onev;

		onev.idxOne[0] = SZ_MTX_LOW_FIRST(this);
		if(onev.idxOne[0] != SZ_MTX_LOW_FIRST(s_mtx) || onev.idxOne[0] != SZ_MTX_LOW_FIRST(r_mtx))
			throwFault(-1, "size inconsistance\n");

		mone(tcxt ? tcxt : mxTcr->trcCxt(r_mtx->didground), s_mtx, r_mtx, 2, &onev, AOP_TYPE1, DIAGO_MUL, PDN, 0);
	}
	void mdiag_fill(TContext *tcxt, Matrixr *r_mtx)
	{
		OneVar onev;

		onev.idxOne[0] = SZ_MTX_LOW_FIRST(this);
		if(onev.idxOne[0] != SZ_MTX_LOW_FIRST(r_mtx)) throwFault(-1, "size inconsistance\n");

		mone(tcxt ? tcxt : mxTcr->trcCxt(r_mtx->didground), nullx, r_mtx, 0, &onev, AOP_TYPE1, DIAGO_FILL, PDN, 0);
	}
};