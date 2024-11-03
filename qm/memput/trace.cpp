
#include "trace.h"
#include "matrix.h"
#include "misc/fio.h"
#include "baper.h"

void Capsule::allocShadow(intt sz, intt gid)
{
	/*if(vcaps->ds && sz <= ((ShadowCap *)vcaps->ds)->shadowsz) {
		dataShadow = (ShadowCap *)vcaps->ds;
		gradShadow = (ShadowCap *)vcaps->gs;
		return;
	}*/
	dbgIdShadow = gid;
	dataShadow = DMATFX(vcaps)->shadowAlloc(sz, gid);
	gradShadow = GMATFX(vcaps)->shadowAlloc(sz, gid);
	//vcaps->ds = dataShadow;
	//vcaps->gs = gradShadow;
}
void Capsule::freeShadow(void)
{
	/*if(vcaps->ds == nullptr) {
		dataShadow = gradShadow = nullptr;
		return;
	}*/
	dbgIdShadow = -1;
	DMATFX(vcaps)->shadowFree(dataShadow->didshadow);
	GMATFX(vcaps)->shadowFree(gradShadow->didshadow);
	//vcaps->ds = nullptr;
	dataShadow = gradShadow = nullptr;
}
void Capsule::setShadow(intt gid, bool grad)
{
	if(grad) {
		if(gid != gradShadow->didshadow) throwFault(-1, "set shadow inconsistant gid\n");
		GMATFX(vcaps)->shadowSet(gradShadow->devShadow, gradShadow->didshadow);
	} else {
		if(gid != dataShadow->didshadow) throwFault(-1, "set shadow inconsistant gid\n");
		DMATFX(vcaps)->shadowSet(dataShadow->devShadow, dataShadow->didshadow);
	}
}
Trace::~Trace()
{
	VectorTag *vtag = vtLast;

	for(;vtag; vtag = vtLast) {
		vtLast = vtag->ptrPrev;
		delete vtag->pvector;
	}
	for(;mtxlist; mtxlist = mtxlist->ptrRight) {
		mtxlist->mtxrm(true);
	}
	NameScope *nsc = nsOrder;
	for(;nsc; nsc = nsc->ptrRight2) {//가중치 저장
	}
	CLOSE_MUT_(mutTrc);
	rsc::rAllocator(hmalc);

	CLOSE_MUT_(mutMerge);
	freeBaper();
	CLOSE_MUT_(mutArrange);
	for(Trace *trc = lstSlav;trc; trc = lstSlav) {
		lstSlav = trc->ptrRight;
		delete trc;
	}
	tcxtarr->rmTCxtArray();
}
void Trace::listm(Matrixr *mx)
{
	APPEND_LIST(mtxlist, mx);
}
//axid는 매트릭스 생성내부에서 포인터로 설정되므로 axid의 life cycle이 매트릭스와 같아야 한다.
Matrixr *Trace::instMatrix(Matrixr *mtx, ubytet qt, intt ndim, intt *axid, bool o_mut, intt gid, Matrixr *mast)
{
	if(mtx) {//아래조건에서 마스터 포인터가 주어졌을때는 무조건 재설정 실행은 frsz)의 설명 참조
		if(mtx->resizeing(ndim, axid, gid) == false || mast) mtx->amxm(ndim, axid, 0, gid, mast);
	} else {
		switch(qt) {
		case NONET_TP:
			break;
		case BYTET_TP:
			break;
		case SHORTT_TP:
			break;
		case FLOATT_TP:
			mtx = new((Tracer *)this)Matrix<floatt>(this, qt, ndim, axid, o_mut, gid, (Matrix<floatt> *)mast);
			break;
		case INTT_TP:
			mtx = new((Tracer *)this)Matrix<intt>(this, qt, ndim, axid, o_mut, gid, (Matrix<intt> *)mast);
			break;
		case LONGT_TP:
			break;
		case DOUBLET_TP:
			break;
		}
		mtx->npAbrib = npAbrib;
		mtx->lapType = lapType;
	}
	return mtx;
}
#ifdef OPT_WIN
#include <AccCtrl.h>
#include <aclapi.h>
intx xcreate_directory(bytex *name)
{
	SECURITY_ATTRIBUTES sa;
	SECURITY_DESCRIPTOR sd;
	PSID pEveryoneSID = NULL;
	SID_IDENTIFIER_AUTHORITY SIDAuthWorld = SECURITY_WORLD_SID_AUTHORITY;
	EXPLICIT_ACCESS ea[2];
	PACL pacl = NULL;

	if(!AllocateAndInitializeSid(&SIDAuthWorld, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, &pEveryoneSID)) return -1;

	ZeroMemory(&ea, 2 * sizeof(EXPLICIT_ACCESS));
	ea[0].grfAccessPermissions = GENERIC_ALL;
	ea[0].grfAccessMode = SET_ACCESS;
	ea[0].grfInheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT;
	ea[0].Trustee.TrusteeForm = TRUSTEE_IS_SID;
	ea[0].Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	ea[0].Trustee.ptstrName = (LPTSTR)pEveryoneSID;

	SetEntriesInAcl(1, ea, NULL, &pacl);
	InitializeSecurityDescriptor(&sd, SECURITY_DESCRIPTOR_REVISION);
	SetSecurityDescriptorDacl(&sd, TRUE, pacl, FALSE);
	//SetSecurityDescriptorSacl(&sd, TRUE,pacl , FALSE);

	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.lpSecurityDescriptor = &sd;
	sa.bInheritHandle = TRUE;

	wchar_t s[1024];
	mbstowcs(s, name, strlen(name) + 1);
	CreateDirectory(s, &sa);

	if(pEveryoneSID) {
		FreeSid(pEveryoneSID);
	}
	if(pacl) {
		LocalFree(pacl);
	}
	return 0;
}
//success - 0, fail - -1, exist - 1
intx XFileMakePath(bytex *path)
{
	bytex DirName[256];
	bytex *p = path;
	bytex *q = DirName;

	if(XFileExist(path)) return 1;

	while(*p) {
		if(('\\' == *p) || ('/' == *p)) {
			if(':' != *(p - 1)) {
				if(xcreate_directory(DirName) < 0) return -1;//if(CreateDirectory(DirName, NULL) < 0) return -1;
			}
		}
		*q++ = *p++;
		*q = '\0';
	}
	if(xcreate_directory(DirName) < 0) return -1;//if(CreateDirectory(DirName, NULL) < 0) return -1;

	return 0;
}
#else
//#include <stdio.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <dirent.h> 
//#include <string.h> 
#include <sys/stat.h>
//success - 0, fail - -1, exist - 1
intx XFileMakePath(bytex* path)
{
	char szdirname[FILE_PATH_LEN];
	DIR *dirp;
	int ipos = 0;

	if(XFileExist(path)) return 1;

	if(*path == '/') {
		szdirname[ipos++] = *path++;
	}
	umask(0000);
	while(*path != '\0') {
		if(*path == '/') {
			szdirname[ipos] = '\0';
			dirp = opendir(szdirname);
			if(dirp) {
				szdirname[ipos] = '/';
				closedir(dirp);
			} else {
				if(mkdir(szdirname, 0777)) return -1;
			}
		}
		szdirname[ipos++] = *path;
		path++;
	}
	//폴더명 끝에 '/' 로 끝날 수도 있고 그렇지 않을 수도 있으므로... 깔끔하게
	if(szdirname[ipos - 1] != '/') {
		szdirname[ipos] = '\0';
		if(mkdir(szdirname, 0777)) return -1;
	}
	return 0;
}
#endif
void Trace::saveWeight(void)
{
	filex wfp;
	bytet name[NAME_LENG], *data;
	intt off = 0, size;

	if(strlen(rPath) > 1) XFileMakePath(rPath);

	sprintf(name, "%s/%s.dfx", rPath, nsOrder->nsName);
	XFileOpen(name, wfp);
	if(XFileFailOpen(wfp)) throwFault(-1, "file open fail\n");
	try {
		for(Capsule *cap = persistw();cap; cap = cap->ptrRight, off += size) {//oiro.
			if(printLoad) {
				if(cap->vcaps->fxName) printf("%s\n", cap->vcaps->fxName);
				cap->vcaps->printo();
			}
			data = (bytet *)cap->vcaps->begin_p();
			size = cap->vcaps->sizefx();
			XFileSet(wfp, off);
			XFileWrite(wfp, data, size);
		}
		XFileFlush3(wfp);
		XFileClose(wfp);
	} catch(ExptObject eo) {
		XFileClose(wfp);
		throwFault(-1, eo.msg);
	}
}
intt Trace::loadWeight(void)
{
	Capsule *cap;
	filex wfp;
	bytet name[NAME_LENG], *data;
	intt off = 0, size;

	if(loadedWeight) return 1;

	sprintf(name, "%s/%s.dfx", rPath, nsOrder->nsName);
	XFileOpen2(name, wfp);
	if(XFileFailOpen(wfp)) return 0;
	try {
		for(cap = persistw();cap; cap = cap->ptrRight, off += size) {//oiro.
			data = (bytet *)cap->vcaps->begin_wp();
			size = cap->vcaps->sizefx();
			XFileSet(wfp, off);
			XFileRead(wfp, data, size);
			if(printLoad) {
				if(cap->vcaps->fxName) printf("%s\n", cap->vcaps->fxName);
				cap->vcaps->printo();
			}
		}
		XFileClose(wfp);
		loadedWeight = 1;
	} catch(ExptObject eo) {
		XFileClose(wfp);
		throwFault(-1, eo.msg);
	}
	return 1;
}
void Trace::truncWeight(void)
{
	bytet name[NAME_LENG];

	sprintf(name, "%s/%s.dfx", rPath, nsOrder->nsName);
	XFileRemove(nsOrder->nsName);
}
void Trace::printWeight(void)
{
	for(Capsule *cap = persistw();cap; cap = cap->ptrRight) {//oiro.
		if(cap->vcaps->fxName) printf("%s\n", cap->vcaps->fxName);
		cap->vcaps->printo();
	}
}
Flux *Trace::tcr_combination(Flux *in, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, bool one_batch)
{
	hbaper->bcombination(in, width, stride, exc_contig_r, zero_pading, one_batch);
	Flux *expand_mask = rsc::rsc_combination(in, width, stride, exc_contig_r, zero_pading, 0, one_batch);
	return expand_mask;
}
Flux *Trace::tcr_combination2(Flux *in, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, bool one_batch)
{
	hbaper->bcombination2(in, width, stride, exc_contig_r, zero_pading, one_batch);
	//apply를 생성하여 학습/평가 타임에 포워드 실행 되게 한다.
	ApCombination *apcb = new(this)ApCombination(this, in, width, stride, exc_contig_r, zero_pading, one_batch);

	in->backend_ps(in, nullptr, apcb->apOut, apcb);
	return apcb->apOut;
}
void Trace::directx(bool on)
{
	directExec = on;
	hbaper->inner_flux = on;
}
