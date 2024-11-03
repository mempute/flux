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
				CudaDevSet(didground);//메모리 해제는 생성 디바이스에 포커스하여 수행한다. 바로 뒤에
							//쓰레드와 공유되는 디바이스에 할당되므로 해제할때 포커스 변경되도 상관없다.
				cudaerror(cudaFree(devGround), "matrix cuda free error");
				cudaDeviceSynchronize();
				devGround = nullptr;
			}
			if(r_mut) {//모두 삭제
				shadowFree();
				lshadow = nullptr;
				if(devDock) {//배치 분할 분산 실행할때 그래디언트 값을 슬레이브에서  
					//printf("cuda dock free %p\n", devDock);
					CudaDevSet(didground);//마스터로 sum할때 슬레이브 그라운드가 다르면 쉐도우잉
					cudaerror(cudaFree(devDock), "matrix cuda free error");//하기위해 할당했던 것 해제
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
	void amxm(intt ndim, intt *axid, sytet init, intt gid = -1, Matrixr *mast = nullptr) //ground 할당에서만 호출됨
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
			if(gid == JUST_SIZEDEF) return;//사이즈 계산 할수있게만 초기화하고 이후 진행안함, 나중에 리사이즈에서 메모리 할당
			if(mastmat) {//reshape인 경우 reshape되는 플럭스의 메모리를 공유한다.쉐도우는 동일하게 처리
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
				if(init) {//초기 빌드시점에 할당할때는 장비 변경없이 메인 쓰레드에서만 할당되고 모자르면 
					size_t sz_free = getmfreegpu(didground);//메모리 할당하지 않는다.
					if(mxsz * sizeof(DT) + mxTcr->DEV_AVAIL_MARGIN >= sz_free) {//그레프에 의한 실행단계
						mxTcr->prompt = 0;//에서는 사이즈 체크를 하여 할당하므로 실패할일 없고 빌드단계
						//printf("xxx %d %d %d\n", sz_free, didground, mxsz);
						return;//에서 할당일 경우만 여기서 체크되어 할당실패이면 즉시실행을 리셋하여 
					}			//빌드단계에서 오퍼레이션이 실행되지 않게 하고 빌드만 되게 한다.
				} else {//그래프 수행 쓰레드와 장비 공유된다.그래프 수행 쓸레드에서 할당 메모리
					CudaDevSet(didground);//계산하여 메모리 모자르는 경우 없다.
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

			CudaDevSet(didground);//기존 메모리 해제
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
	//이하 두개 함수는 플럭스의 그라운드 메모리와 별개로 할당되므로 mastmat reshape된 플럭스도 동일하게 적용
	void dockAlloc(void)
	{
		if(devDock) return;
		intt sz = MTX_SIZE(this) * sizeof(DT);
		CudaDevSet(didground);//슬레이브 실행에서 사용할 메모리는 그라운드와 같은 기계위치이다.
								//슬레이브 수행 쓰레드와 장비 공유된다.
		cudaError_t error = cudaMalloc((void**)&devDock, sz);
		cudaerror(error, "dock alloc error");
		cudaMemset(devDock, 0x00, sz);
		cudaDeviceSynchronize();
	}
	void copyHostToGround(void)
	{
		//가중치일경우 초기될때 호스트 메모리로 수행되고 초기화 된후 한번 가중치를 처음 
		//엑세스 하는 곳에서 호스트->디바이스로 복사된다. 그런데 수행전에 그래프쓰레드에 의해 쉐도우
		//메모리 설정될 경우에 쉐도우 메모리에 호스트의 초기화 내용이 복사되고 그라운드 메모리는 널값
		//상태이므로 두번째 수행부터는 널값의 그라운드가 쉐도우에 복사되어 가중치 값이 사라지게 된다.
		//따라서 초기 쉐도우 할당시점(가중치 값은 이미 호스트 메모리에 설정됨)에 그라운드에 값을 설정한다.
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
	intt sizem2(intt n, bool bsize = 1) //배치 n개로 했을때의 사이즈
	{
		intt sz = MTX_SIZE_RK(this, 1);

		return sz * n * (bsize ? sizeof(DT) : 1);
	}
	void inCopy(Matrixr *i_mtx, sytet gpu) //포커스된 디비이스 메모리는 그래프 쓰레드에서 focusDevice에서 고려되었다
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
	intt copyMemory(void *pm, sytet gpu, intt begin, intt size)//포커스된 디비이스 메모리는 그래프 쓰레드에서 focusDevice에서 고려되었다
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
		cpmhot = 1;//이후 호스트 메모리에 쓰여질 것이므로
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
			if(n) {//n이 주어졌으면 irank의 하나 하위 랭크의 사이즈에 n을 곱해 irank의 n까지의 사이즈를 구한다.
				intt sz = (irank >= mxndim ? 1 : MRANK_SIZE(mxranksz, (irank + 1)));
				if(n * sz < *rsz) *rsz = n * sz;//사이즈가 더 적을 경우만 재설정, 최대 irank를 넘지않게
			}
		}
		return (mxmHost + off);
	}
	void write_p(intt xid[], void *dat, intt irank, intt wsz = 0)
	{
		intt off;

		CopyDeviceToHost(this, 0);
		_idx2offset(mxndim, mxranksz, xid, off);
		intt sz = (irank >= mxndim ? 1 : MRANK_SIZE(mxranksz, irank));//적재할 랭크의 사이즈
		memcpy(mxmHost + off, dat, (wsz ? (wsz > sz ? sz : wsz) : sz) * sizeof(DT));//wsz - 적재할 랭크의 디멘젼 전체 적재할 사이즈
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
		mone(tcxt, nullx, this, 0, &onev, AOP_RANDOM, RAND_T_N, PDC3, 0);//gpu실행은 사이즈가 크면 에러나므로 cpu로 실행하게
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
		mone(tcxt, nullx, this, 0, &onev, AOP_RANDOM, RAND_T_U, PDC3, 0);//gpu실행은 사이즈가 크면 에러나므로 cpu로 실행하게
	}
	intt uniform(TContext *tcxt, Univ *uni)
	{//여기 함수들은 모두 초기값 설정하는 것이므로 이전 cpu/gpu메모리 고려 않는다.
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

			CopyDeviceToHost(src, 0);//호스트 메모리에서 값을 읽어야 하므로 동기화
			if(src->mxndim <= axis) {//맨 하위 차원 확장
				intt sn = MTX_SIZE(src);
				DT a;
				for(intt i = 0;i < sn; i++) {//소스의 원소 한 개씩을
					a = *(src->mxmHost + i);
					for(intt j = 0;j < n; j++) *(mxmHost + off++) = a;//타겟에 n개씩 중복하여 복사한다.
				}
			} else {
				intt sz = MTX_SIZE_RK(src, axis);
				intt sn = MTX_SIZE(src) / sz;//소스의 차상위 사이즈로 전체 사이즈를 나누어 최상위 디멘젼 산출
				DT *p;
				for(intt i = 0;i < sn; i++) {
					p = src->mxmHost + i * sz;//소스의 맨 하위부터 차상위 디멘젼까지를 
					for(intt j = 0;j < n; j++) {
						memcpy(mxmHost + off, p, sz * sizeof(DT));//타겟에 n개씩 중복하여 복사한다.
						off += sz;
					}
				}
			}
		}
			break; 
		case MIN_MAX_V_OP:
		{
			DT minv, maxv;

			CopyDeviceToHost(this, 0);//호스트 메모리에서 값을 읽어야 하므로 동기화
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

			CopyDeviceToHost(this, 0);//호스트 메모리에서 값을 읽어야 하므로 동기화
			for(intt i = 0; i < len; i++) mean += *(mxmHost + i);//입력값 합계
			mean /= len;
			for(intt i = 0; i < len; i++) {
				d = (*(mxmHost + i) - mean);//평균 편차
				stv += d * d;//분산, 평균편차 제곱 합
			}
			std = 1.0 / std::sqrt(stv / len + 1e-9);//표준편차(분산평균) 역수
			for(intt i = 0; i < len; i++) {//입력값 평균 편차에 표준편차 역수를 곱하여 표준값 적재
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
			
			if(avg_pre || origin) {//이전 시퀀스의 피쳐별 평균을 prev값으로 이번 시퀀스의 피쳐별 지니계수 산출
				if(mxndim < 2) throwFault(-1, "dims small\n");
				szseq = SZ_MTX_LOW_SECOND(this);
				nseq = szseq / nfeat;
				if(avg_pre) {
					psum = (DT *)malloc(nfeat * sizeof(DT));
					for(intt i = 0; i < nfeat; i++) *(psum + i) = 0;
				}
			}
			for(intt i = 0; i < nfeat; i++) *(prev + i) = 0.5;//초기 prev값으로 0.5

			CopyDeviceToHost(this, 0);//호스트 메모리에서 값을 읽어야 하므로 동기화
			if(origin) CopyDeviceToHost(origin, 0);
			for(intt i = 0; i < len; i++) {
				if(reverse) {//지니계수값으로부터 원래값 복원
					gv = *(mxmHost + i);
					//음수와 0이 있으면 전체 이동하여 양수로 만든다. (prev가 0값이면 지니계수가
					//1이되어 복원이 안된다, 따라서 데이터를 양수로 변환)
					curv = ((*(prev + i % nfeat) + 1e-9) * gv) / ((1 - gv) + 1e-9);
					*(mxmHost + i) = curv;
					//printf("%d:%d %f %f %f\n", i / nfeat, i % nfeat, *(prev + i % nfeat), gv, curv);
				} else {//데이터로부터 지니계수 기울기 산출
					curv = *(mxmHost + i);
					*(mxmHost + i) = curv / (*(prev + i % nfeat) + curv);//지니계수 기울기
					//printf("%d:%d %f %f %f %f\n", i / nfeat, i % nfeat, *(prev + i % nfeat), curv, *(mxmHost + i), *(prev + i % nfeat) + curv);
				}
				if(avg_pre) {//시퀀스 단위로 이전값 설정
					if(origin) curv = *(origin->mxmHost + i);//복원과정이고 오리진이 주어졌으면
					*(psum + i % nfeat) += curv;//복원값을 이전값 설정	//오리진에서 이전값 설정
					intt j = i + 1;
					if(j >= szseq) {//첫번째 시퀀스는 초깂을 이전값으로 하므로 두번째 시퀀스부터 이전값 설정.
						if((j / nfeat) % nseq == 0) {//이전시퀀스의 마지막 아이템 마지막 피쳐에서
							//시작하여 다음 시퀀스의 첫번째 아이템 동안 펴쳐별 이전 시퀀스 평균값을
							*(prev + j % nfeat) = *(psum + j % nfeat) / nseq;//계산하여
							*(psum + j % nfeat) = 0;						//이전값 설정
						}
					} else *(prev + i % nfeat) = curv;//첫번째 시퀀스는 피쳐별로 이전 아이템값을 이전값으로 설정.
				} else if(origin) {//아이템단위 이전값 설정이고 복원과정이고 오리진이 주어졌으면
					intt j = i + nfeat;//이전 시퀀스의 마지막 아이템 동안 피쳐별로 오리진의
					if(j >= szseq && (j / nfeat) % nseq == 0) {//마지막 아이템 값을 이전값으로
						//설정, 즉 시퀀스단위로 첫번째 아이템만 오리진의 이전 시퀀스의 마지막 아이템값을
						*(prev + i % nfeat) = *(origin->mxmHost + i);//이전값으로 설정.
					} else *(prev + i % nfeat) = curv;//그 이외엔 이번값을 다음의 이전값으로 설정
				} else {//피쳐별로 이번 아이템의 값을 다음 아이템의 이전값으로 설정.(복원과정이면
					*(prev + i % nfeat) = curv;//이번 복원값이고 계수 산출값이면 이번 데이터값)
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
					if(leaf_one && *(mxshape + mxndim - 1) == 1) j = 2;//맨끝(하위) 차원이 1이면 개행하지 않고 옆으로 나열
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
				if(leaf_one && *(mxshape + mxndim - 1) == 1) j = 2;//맨끝(하위) 차원이 1이면 개행하지 않고 옆으로 나열
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
					if(leaf_one && *(mxshape + mxndim - 1) == 1) j = 2;//맨끝(하위) 차원이 1이면 개행하지 않고 옆으로 나열
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
		} else {//이 케이스는 trans order필요없다.(축을 지정하여 곱을 하는 기능이 더 포괄적이므로) 
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
		if(parv == nullx) {//역전파 없이 내부에서 단순 산술 연산 호출
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
				if(mtx->nbackw > 1) b = 1;//한개만 백워드 멀티여도 모두 일치시킨다.
				*((DT **)dev_mptr + i) = mtx->mxmDevice;
				*((DT **)host_mptr + i) = mtx->mxmHost;
				//printf("%p\n", mtx->mxmDevice);
			}
		} else if(gpu && mxshape[axis] < 30) {//분할 차원 디멘젼이 30개 이하아면 분할차원 매 원소마다 인덱스를 설정해 바로 찾게한다.
			intt inner_sz = (axis == mxndim - 1 ? 1 : MRANK_SIZE(mxranksz, axis + 1));
			intt k = 0;
			dev_mptr = tcxt->cxalloc(mxshape[axis] * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(mxshape[axis] * sizeof(void *), ccv->mptrHostSecondary);
			intt *sdim = tcxt->cxalloc(mxshape[axis] * sizeof(intt), ccv->sdimCat);
			intt *sbase = tcxt->cxalloc(mxshape[axis] * sizeof(intt), ccv->sbaseCat);
			intt sbaseoff = 0;
			for(i = 0;i < nsplit; i++) {
				mtx = *((Matrix<DT> **)psplit_mtx + i);
				if(mtx->nbackw > 1) b = 1;//한개만 백워드 멀티여도 모두 일치시킨다.
				for(intt j = 0;j < mtx->mxshape[axis]; j++, k++) {
					*((DT **)dev_mptr + k) = mtx->mxmDevice;
					*((DT **)host_mptr + k) = mtx->mxmHost;
					*((intt *)sdim + k) = mtx->mxshape[axis] * inner_sz;
					*((intt *)sbase + k) = sbaseoff;
				}
				sbaseoff += mtx->mxshape[axis] * inner_sz;
			}
			each_map = 1;
		} else {//분할 차원 디멘젼이 30개 보다크면 분할 갯수별로 인덱스를 설정한다.
			intt inner_sz = (axis == mxndim - 1 ? 1 : MRANK_SIZE(mxranksz, axis + 1));
			dev_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(nsplit * sizeof(void *), ccv->mptrHostSecondary);
			intt *sdim = tcxt->cxalloc(nsplit * sizeof(intt), ccv->sdimCat);
			intt *sbase = tcxt->cxalloc(nsplit * sizeof(intt), ccv->sbaseCat);
			intt sbaseoff = 0;
			for(i = 0;i < nsplit; i++) {
				mtx = *((Matrix<DT> **)psplit_mtx + i);
				if(mtx->nbackw > 1) b = 1;//한개만 백워드 멀티여도 모두 일치시킨다.
				*((DT **)dev_mptr + i) = mtx->mxmDevice;
				*((DT **)host_mptr + i) = mtx->mxmHost;
				*((intt *)sdim + i) = mtx->mxshape[axis] * inner_sz;
				*((intt *)sbase + i) = sbaseoff;
				//printf("%p %d %d\n", mtx->mxmHost, mtx->mxshape[axis] * inner_sz, sbaseoff);
				sbaseoff += mtx->mxshape[axis] * inner_sz;
			}
			each_map = 0;
		}
		if(b == 0) bw = 0;//모두 봭워드 멀티 참조가 아니면 백워드 플래그를 리셋시킨다.split연산에서 타겟 참조없이 쓰기만되므로
		tcxt->syncCxt2Dev(gpu);

		if(gpu) {
			CopyHostToDevice(this, 0);
			if(bw) {//백워드 실행이고 한개라도 봭워드 멀티 참조이면 전채 split매트릭스를 cpu,gpu 메모리를 일치시킨다.
				for(i = 0;i < nsplit; i++) {
					mtx = *((Matrix<DT> **)psplit_mtx + i);
					CopyHostToDevice(mtx, 0);
				}
			}
		} else {
			CopyDeviceToHost(this, 0);
			if(bw) {//백워드 실행이고 한개라도 봭워드 멀티 참조이면 전채 split매트릭스를 cpu,gpu 메모리를 일치시킨다.
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
		if(parity) {//분할 사이즈가 균등인 것을 합침
			dev_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrDevSecondary);
			host_mptr = tcxt->cxalloc(ncat * sizeof(void *), ccv->mptrHostSecondary);
			for(i = 0;i < ncat; i++) {
				mtx = *((Matrix<DT> **)pcat_mtx + i);
				*((DT **)dev_mptr + i) = mtx->mxmDevice;
				*((DT **)host_mptr + i) = mtx->mxmHost;
			}
		} else if(gpu && mxshape[axis] < 30) {//분할 차원 디멘젼이 30개 이하아면 분할차원 매 원소마다 인덱스를 설정해 바로 찾게한다.
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
		} else {//분할 차원 디멘젼이 30개 보다크면 분할 플럭스 갯수별로 인덱스를 설정한다.
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
		if(nbackw <= 1) bw = 0;//타겟이 봭워드 멀티 참조가 아니면 백워드 플래그를 리셋시킨다.concat연산에서 타겟 참조없이 쓰기만되므로
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
		if(rmtx->nbackw <= 1) bw = 0;//타겟이 봭워드 멀티 참조가 아니면 백워드 플래그를 리셋시킨다.
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
		SyncHstDev(gpu, 1, this, ((Matrix *)sum_mtx), ((Matrix *)max_mtx));//max_mtx가 리턴매트릭스가 아닌 읽기참조되는 것이므로 rcopy 1로 설정
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
		bool gpu = mxRutra->policyTrack(MTX_SIZE(this), 0, feat_sz, PDG, width, n, mxTcr->cpmMode);//rmtx는 this에서 feat_sz가 차원 축소된 매트릭스
		
		rmtx->resetMemory(gpu ? 1 : -1);
		SyncHstDev(gpu, 1, this, ((Matrix *)tar_mtx), rmtx);//바로위에서 리셋한 rmtx를 반영하기위해 rcopy 1로 설정.
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
			gpu = mxRutra->policyTrack3(MTX_SIZE(this), width, n, mean, mxTcr->cpmMode);//전체 합 평균을 구한는 것이므로 cpu는 분할처리 안하게
			rmtx->resetMemory(gpu ? 1 : -1);
		} else gpu = mxRutra->policyTrack(MTX_SIZE(this), -1, nullx, PDG, width, n, mxTcr->cpmMode);
		SyncHstDev(gpu, 1, this, tmtx, rmtx);//바로위에서 리셋한 rmtx를 반영하기위해 rcopy 1로 설정.
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
		SyncHstDev(gpu, 1, this, ((Matrix *)0), rmtx);//바로위에서 리셋한 rmtx를 반영하기위해 rcopy 1로 설정.
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
		SyncHstDev(gpu, 1, this, ((Matrix *)0), rmtx);//수행될깨 rmtx가 읽기 참조되므로 rcopy 1
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
		//랜덤생성과 같이 반드시 gpu로 처리해야 하는 케이스이면(PDC3) pdiv를 1로 설정.
		bool gpu = mxRutra->policyTrack(sz, 0, feat_sz, pdiv, width, n, pdiv == PDC3 ? 1 : mxTcr->cpmMode);
		if(pvar) {
			void *p;
			tcxt->cxbegin();
			p = tcxt->cxalloc(sizeof(OneVar), i);
			memcpy(p, pvar, sizeof(OneVar));
			tcxt->syncCxt2Dev(gpu);
		}
		if(rplus > 1) rmtx->resetMemory(gpu ? 1 : -1);
		else if(rplus > 0 && rmtx->nbackw <= 1) rplus = 0;//rplus가 음수이면 무조건 일치
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
		//ga, be, g_ga, g_be는 가중치이므로 run과정에서 초기화 됨.
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