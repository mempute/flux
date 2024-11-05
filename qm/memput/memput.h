#pragma once

#include <memory>
#include <iostream>
#include <vector>
using namespace std;

namespace memput {
	namespace mp {
		typedef signed char sytet;
		typedef char bytet; // 1 byte
		typedef short shortt; // 2 byte
		typedef int intt; // 4 byte
		typedef unsigned char ubytet;
		typedef unsigned short ushortt;
		typedef unsigned int uintt;
		typedef float floatt;
		typedef double doublet;

#ifdef OPT_WIN
		typedef __int64 longt;
		typedef unsigned __int64 unitt;
#else
		typedef long long longt;
		typedef unsigned long long unitt;
#endif
		class Flux;
		typedef struct FxAnchor_ {
			Flux *fxPoint;
			struct FxAnchor_ *ptrLeft, *ptrRight;
		} FxAnchor;
		typedef struct WeightList_ {
			Flux *weightFx;
			struct WeightList_ *ptrLeft, *ptrRight;
		} WeightList;
		typedef struct NameScope_ {
			intt idNameFx;
			bool reuseScope;
			bytet *nsName;
			WeightList *nameWeights, *anonymWeights;
			struct NameScope_ *ptrParent, *ptrChild, *ptrLeft, *ptrRight, *ptrLeft2, *ptrRight2;
		} NameScope;//그래프 빌그 과정에서 생성

		class Tracer {
		public:
			bytet characterName[24];
			intt characterType, tcr_reserve;
			sytet dbgStep, dbgStep2;
			virtual ~Tracer() {}
			virtual NameScope *namescope(const bytet *nsm, bool reuse = 0) = 0;
			virtual NameScope *namescope(intt i, bool reuse = 0) = 0;
			virtual void setmsg(bytet *msg) = 0;
			virtual void endscope(void) = 0;
			virtual void init_train(void) = 0;
			virtual void run(Flux *target) = 0;
			virtual void run(vector<Flux *> target) = 0;
			virtual void run(vector<Flux *> *target) = 0;
			virtual void reposet(bytet *rpath) = 0;
			virtual void npset(intt n) = 0;
			virtual void lapset(sytet lap) = 0;
			virtual void gprset(floatt d) = 0;
			virtual void modeset(sytet cpm) = 0;
			virtual void multiDevice(vector<intt> axid) = 0;
			virtual void portingGraph(Tracer *target) = 0;
			virtual Flux *getFlux(Flux *sfx) = 0;
			virtual void promptMode(bool on) = 0;
			virtual void sizeBatch(intt sz) = 0;
			virtual void saveWeight(void) = 0;
			virtual intt loadWeight(void) = 0;
			virtual void truncWeight(void) = 0;
			virtual void printWeight(void) = 0;
			virtual void setgpudev(intt gid) = 0;
			virtual void traceopt(intt i, doublet v) = 0;
			virtual void directx(bool on) = 0;
			virtual void setbygate(Flux *gate, intt nout, Flux *go, Flux *end, intt embedim = -1, intt latent_sz = -1, intt indiscret = -1) = 0;
			virtual NameScope *findnsc(bytet *nsm, sytet root) = 0;
			virtual vector<Flux *> *trainvar(NameScope *nsc) = 0;
			void characterSet(bytet *char_name, intt char_type)
			{
				strcpy(characterName, char_name);
				characterType = char_type;
			}
		};
		Tracer *trace(sytet stepw = 0, const bytet *name = nullptr);

#define GEN_T_FLUX	0
#define GEN_T_TACT	1
#define GEN_T_CAP	2

		class Typer {
		public:
			void *operator new(size_t size, Tracer *tcr);
		};

		class Contact : public Typer {
		public:
			sytet Tgener = GEN_T_TACT;
			Contact *ptrLeft, *ptrRight;
			Typer *vcontact;
		};

		class Univ {
		public:
			sytet opvuni;
			sytet tpvuni;
			longt cvuni, cvuni2;
			Univ(sytet op, sytet tp)
			{
				opvuni = op; tpvuni = tp;
			}
		};
#define T_XAVIER_INIT	0
#define T_HE_INIT		1
#define T_ONE_INIT		2
#define T_ZERO_INIT		3
		class Initializer {
		public:
			static intt xavier(Flux *fx);
			static intt he(Flux *fx);
			static intt one(Flux *fx);
			static intt zero(Flux *fx);
		};
#define MX_DIM	10 //1개를 시스템에서 사용하므로 실제 사용은 이보다 1개 적어야 한다.
#define TON		0
#define TOA		1
#define TOB		2
#define TOT		3
#define AOP_MUL		0
#define AOP_PLUS	1
#define AOP_DIV		2
#define AOP_MINUS	3

#define ACTF_TANH	0
#define ACTF_RELU	2
#define ACTF_SIGM	4
#define ACTF_LRELU	6
#define ACTF2_PRELU	100

#define DEFA_LRELU	0.01
		typedef intt(*vinitfp)(Flux *);

		class Flux : public Typer {
		public:
			sytet Tgener = GEN_T_FLUX;
			ubytet qType, fxType;
			intt nRefer, ibRefer, nbRefer;
			Typer *bwAp, *bwLink, *directLink;
			Contact *fwAp;
			void *quantum;
			intt fdim, fxSize;
			intt fshape[MX_DIM];
			bytet *fxName;
			Tracer *fxTcr;
			intt backwv, ofxArrange;
			//intt nJoint;
			intt scaleout;
			intt bregid;
			bool termifx;
			bool meanAfter;
			bool trainherit;
			bool bwbreak;
			bool partitionfx;
			bool changeGround;
			vinitfp vinitter;
			void *cursorP;
			Flux *ptrLeft, *ptrRight;//lstTarget사용
			Flux *ptrMastfx;
			//void *ds, *gs;

			void init(Tracer *tcr, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name);
			intt sizef(void);
			intt sizefx(bool t_size = 0);
			intt sizefx2(intt n, bool bsize = 1);
			Flux(Tracer *tcr, intt *axid, intt ndim, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
			Flux(Tracer *tcr, initializer_list<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
			Flux(Tracer *tcr, vector<intt> &axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr, Flux *mast = nullptr);

			void instTens(bool inst, intt gid = -1, Flux *mast = nullptr);
			void fdims(intt ndim, intt *pdim);
			void fdims(initializer_list<intt> axid);
			bool checkInBwfx(bool lock);
			bool checkInBwfx2(void);
			void projected(Typer *ap);
			void referenced(Typer *ap);
			void exec(void *tcxt = 0);
			void backward(void);
			void backend_nstep(Flux *fxp, Flux *fxs, Flux *fxo, void *ap);
			void backend_ps(Flux *fxp, Flux *fxs, Flux *fxo, void *ap, intt vscale = 0);
			void reentrance(bool on);
			void switchTrace(Tracer *tcr, bool on);
			Flux *switchout(Flux *other);
			Flux *arithmetic(Flux *fxs, ubytet qtype, void *sval, sytet arith_op);
			Flux *mul(Flux *fxs, longt sval);
			Flux *plus(Flux *fxs, longt sval);
			Flux *div(Flux *fxs, longt sval);
			Flux *minus(Flux *fxs, longt sval);
			Flux *mul(Flux *fxs, intt sval);
			Flux *plus(Flux *fxs, intt sval);
			Flux *div(Flux *fxs, intt sval);
			Flux *minus(Flux *fxs, intt sval);
			Flux *mul(Flux *fxs, doublet sval);
			Flux *plus(Flux *fxs, doublet sval);
			Flux *div(Flux *fxs, doublet sval);
			Flux *minus(Flux *fxs, doublet sval);
			Flux *mul(Flux *fxs);
			Flux *plus(Flux *fxs);
			Flux *div(Flux *fxs);
			Flux *minus(Flux *fxs);
			Flux *dot(Flux *fxs, vector<vector<intt>> axis, sytet trans_order = 0);
			Flux *matmul(Flux *fxs, sytet trans_order = 0);
			vector<Flux *> *devide(intt axid[MX_DIM], intt ndim, intt nby, intt axis, bool ustack);
			vector<Flux *> *split(intt nby, intt axis);
			vector<Flux *> *unstack(intt axis);
			void resizing5(intt n, intt gid);
			void resizing2(Flux *src, const bytet *msg);
			void resizing6(Flux *src);
			void resizing3(Flux *src);
			void resizing4(intt n);
			void sizeCheck(intt ndim, intt axid[]);
			bool resizing(Flux *src);
			Flux *reshape(vector<intt> axid);
			Flux *bypass(const bytet *msg = nullptr);
			Flux *partition(Tracer *gen_trc = nullptr);
			Flux *adjust(Flux *in);
			Flux *duplicate(Tracer *gen_trc);
			Flux *expand_dims(intt axis);
			Flux *squeeze(intt axis = -1);
			Flux *transpose(vector<intt> axid);
			Flux *softmax(void);
			Flux *squaredDifference(Flux *fxs);
			Flux *softmaxCrossEntropy(Flux *fxt);
			Flux *sum(bool batch_sum = 0);
			Flux *mean(bool batch_sum = 0);
			//Flux *rsum(vector<intt> axid, bool mean, bool keep_dims = 0);
			Flux *meanSquareError(Flux *fxt, bool mean = 1);
			Flux *actf(intt actf_op, floatt alpha = DEFA_LRELU);
			Flux *tanh(void);
			Flux *relu(void);
			Flux *lrelu(floatt alpha = DEFA_LRELU);
			Flux *sigmoid(void);
			Flux *prelu(floatt iv);
			Flux *sqrt(void);
			Flux *log(void);
			Flux *embedding_lookup(Flux *fxi);
			Flux *one_hot(intt depth, doublet on_value = 1, doublet off_value = 0, intt axis = -1, intt dtype = -1);
			void boundSlice(intt n, intt code[], intt idx[], bool check);
			Flux *slice(vector<vector<intt>> axis);
			Flux *argmax(intt axis, sytet t_out = -1);
			Flux *vmax(intt axis = -1);
			Flux *equal(Flux *fxs, doublet cmpv, bool cscalr, bool eq);
			Flux *clipValue(doublet low, doublet high);
			bool realwiden(intt n);
			intt copyt(void *psrc, ubytet tsrc, intt bsz);
			void feedt(void *psrc, ubytet tsrc, intt sz);
			intt copyf2(void *pdat, intt bsz, intt begin = 0);
			intt copyf(Flux *src);
			void feedf(void *pdat, intt sz);
			void feedf(Flux *src);
			Flux *feedf(ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
			void dstrw(const bytet dstr[]);
			Flux *overwrite(Flux *tar);
			void resetData(sytet gpu);
			void dumpToGrad(void);
			void resetGrad(void);
			void shape(void);
			void *begin_p(intt off = 0);
			void *begin_wp(intt off = 0);
			void *end_p(void);
			void *read_p(vector<intt> axid, intt *rsz = nullptr, intt n = 0);
			void write_p(vector<intt> axid, void *dat, intt wsz = 0);
			void howrite(Flux *src, intt iseq = 0, intt n = 0);
			void horead(bytet *tar, intt iseq, intt n = 1);
			doublet at_d(intt i);
			doublet at_d2(intt i);
			void printo(sytet leaf_one = 0, sytet width = 1);
			void printg(sytet leaf_one = 0, sytet width = 1);
			void iprinto(intt i = 0, bool nwl = 1);
			void iprintg(intt i = 0, bool nwl = 1);
			Flux *arange(intt len);
			Flux *fill(doublet cv, Flux *fxs = nullptr);
			Flux *fill(floatt cv, Flux *fxs = nullptr);
			Flux *fill(longt cv, Flux *fxs = nullptr);
			void expofill(intt exp_c);
			Flux *expand_elen(intt n, intt axis);
			void _randn(doublet m, doublet v);
			Flux *randn(doublet m, doublet v);
			Flux *randu(doublet m, doublet v);
			Flux *equal(Flux *fxs)
			{
				return equal(fxs, 0, false, true);
			}
			Flux *not_equal(Flux *fxs)
			{
				return equal(fxs, 0, false, false);
			}
			Flux *equal(doublet cmpv)
			{
				return equal(nullptr, cmpv, true, true);
			}
			Flux *not_equal(doublet cmpv)
			{
				return equal(nullptr, cmpv, true, false);
			}
			Flux *layer_dense(intt nout, intt actf_code, vinitfp vfp = Initializer::xavier, const bytet *name = "layer_dense");
			Flux *layer_dense(intt nout, const bytet *actf, vinitfp vfp = Initializer::xavier, const bytet *name = "layer_dense");
			Flux *layer_normal(const bytet *name = "layer_narmal");
			friend Flux *operator+(Flux &a, Flux &b)
			{
				return a.plus(&b);
			}
			friend Flux *operator-(Flux &a, Flux &b)
			{
				return a.minus(&b);
			}
			friend Flux *operator*(Flux &a, Flux &b)
			{
				return a.mul(&b);
			}
			friend Flux *operator/(Flux &a, Flux &b)
			{
				return a.div(&b);
			}
			friend Flux *operator+(Flux &a, doublet f)
			{
				return a.plus(nullptr, f);
			}
			friend Flux *operator-(Flux &a, doublet f)
			{
				return a.minus(nullptr, f);
			}
			friend Flux *operator*(Flux &a, doublet f)
			{
				return a.mul(nullptr, f);
			}
			friend Flux *operator/(Flux &a, doublet f)
			{
				return a.div(nullptr, f);
			}
			friend Flux *operator+(doublet f, Flux &a)
			{
				return a.plus(&a, f);
			}
			friend Flux *operator-(doublet f, Flux &a)
			{
				return a.minus(&a, f);
			}
			friend Flux *operator*(doublet f, Flux &a)
			{
				return a.mul(&a, f);
			}
			friend Flux *operator/(doublet f, Flux &a)
			{
				return a.div(&a, f);
			}
			friend Flux *operator+(Flux &a, longt f)
			{
				return a.plus(nullptr, f);
			}
			friend Flux *operator-(Flux &a, longt f)
			{
				return a.minus(nullptr, f);
			}
			friend Flux *operator*(Flux &a, longt f)
			{
				return a.mul(nullptr, f);
			}
			friend Flux *operator/(Flux &a, longt f)
			{
				return a.div(nullptr, f);
			}
			friend Flux *operator+(longt f, Flux &a)
			{
				return a.plus(&a, f);
			}
			friend Flux *operator-(longt f, Flux &a)
			{
				return a.minus(&a, f);
			}
			friend Flux *operator*(longt f, Flux &a)
			{
				return a.mul(&a, f);
			}
			friend Flux *operator/(longt f, Flux &a)
			{
				return a.div(&a, f);
			}
			Flux *scoopup(intt slidey, intt slidex, intt stridey, intt stridex);
			void scoopup(intt stridey, intt stridex);
			void minmax_normal(doublet minv, doublet maxv);
			void minmax(doublet &minv, doublet &maxv, sytet if_sync = 0);
			void stdnormal(sytet if_sync = 0);
			void xnormal(bool reverse = 0, bool pavg = 0, sytet if_sync = 0);
			void xrnormal(Flux *origin, bool pavg = 0, sytet if_sync = 0);
			void sinpos(intt nseq = 0);// , Flux *fxs = nullptr);
			intt groundid(bool grad);
		};
		//variable은 빌드후에 배치확장되는 것에 사용, constant는 배치확장과 무관한 것에 사용
		const ubytet persistant = 0, trainable = 1, variable = 2, apply = 3, constant = 4, const_apply = 5;//oiro.
		const ubytet tshort = 2, tfloat = 3, tint = 4, tlong = 5, tdouble = 6;

		Flux *flux(Tracer *tcr, initializer_list<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr, intt iname = -1);
		Flux *flux(Tracer *tcr, vector<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Tracer *tcr, Flux *src, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Flux *src, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Tracer *tcr, const bytet dstr[], ubytet qtype=tfloat, ubytet fxtype = constant, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Tracer *tcr, intt ndim, intt axid[], ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);

		class Optimizer : public Typer {
		public:
			intt bregid;
			floatt rLning;
			Optimizer(floatt lr)
			{
				rLning = lr;
			}
			virtual Flux *minimize(Flux *fx, vector<Flux *> *weight_list = nullptr) = 0;
		};
		class AdamOptimizier : public Optimizer {
		public:
			AdamOptimizier(floatt lr) : Optimizer(lr)
			{
			}
			Flux *minimize(Flux *fx, vector<Flux *> *weight_list = nullptr);
		};
		AdamOptimizier *adam_optimizer(Tracer *tcr, floatt lr = -1);

		class GradientDescentOptimizier : public Optimizer {
		public:
			GradientDescentOptimizier(floatt lr) : Optimizer(lr)
			{
			}
			Flux *minimize(Flux *fx, vector<Flux *> *weight_list = nullptr);
		};
		GradientDescentOptimizier *gradient_descent_optimizer(Tracer *tcr, floatt lr = -1);

		class Cell : public Typer {
		public:
			intt trainCount;
			virtual Flux *delegate(Flux *&tgate) { return nullptr; }
			virtual void backtrain(Flux *op_train, Flux *op_loss) {}
			virtual void backpred(Flux *op_pred) {}
			virtual Flux *train(intt *n_train = 0) = 0;
			virtual Flux *predict(Flux **loss_fx = 0) = 0;
			virtual Flux *loss2(void) = 0;
			virtual void accuracy(Flux *&predicts, Flux *&targets, intt discrete_out = -1) = 0;
			virtual Flux *measureAccuracy(void) = 0;
			virtual void recording(void) = 0;
		};
		class Generic : public Cell {
		public:
			intt seq_st[64], ist, in_sz, save_ist, save_insz;
			bool byPolar;
			void *canet;
			NameScope *coaxNsp;
			Flux *zcodec, *dcodec, *clogit, *cypred, *zcodec2, *closs, *bloss, *coptrain, *caccuracy;
			Flux *ctarget, *cpredictions, *copmeasure;
			Generic(void) 
			{ 
				canet = nullptr;
				byPolar = 0; 
			}
			Flux *delegate(Flux *&tgate)
			{
				tgate = ctarget;
				return dcodec;//[batch, y_sz, latent]
			}
			void backtrain(Flux *op_train, Flux *op_loss)
			{
				coptrain = op_train;
				closs = op_loss;
			}
			void backpred(Flux *op_pred)
			{
				clogit = op_pred;
			}
			void makeAnet(Tracer *tcr, intt latent_sz, sytet af);
			void setmhead(Generic *src);
			void recording(void) {}
			NameScope *cbuild(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet step, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
			intt cbuild(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
			intt reduceKernel(Tracer *tcr, intt in_sz);
			NameScope *compress(Tracer *tcr, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, sytet af, floatt lr, const bytet *name = "compress");
			void setFrontendFactor(Generic *from, Flux *_zcodec);
			void setFrontendFactor2(Flux *in);
			NameScope *decompress(Tracer *tcr, intt tar_sz, sytet _endecode = -1, const bytet *name = "decompress");
			NameScope *_coaxial(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet step, sytet af, floatt lr, const bytet *name);
			NameScope *generic(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet step, sytet af, floatt lr, const bytet *name);
			Generic(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
			Generic(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
			Generic(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
			Generic(Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
			void accuracy(Flux *&predicts, Flux *&targets, intt discrete_out = -1);
			Flux *measureAccuracy(void);
			intt reduction(intt outsz, intt szlat = -1);
			void decompose(intt outsz);
			intt decompose2(intt outsz, bool im);
			Flux *cx_optimizer(Flux *loss, floatt lr, sytet opt_t, vector<Flux *> *weight_list = nullptr);
			void connect(Flux *targate, intt outdiscret, sytet step = 0, floatt lr = -1, sytet opt_t = 0);
			Flux *train(intt *n_train = 0);
			Flux *inference(Flux **loss_fx);
			Flux *_predict(Flux *robjec, Flux **loss_fx);
			Flux *predict(Flux **loss_fx = 0);
			Flux *loss2(void) { return closs;  }
		};
		Generic *generic(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
		Generic *stepwise(Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
		Generic *generic(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
		Generic *stepwise(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
		
		class Algol : public Cell {
		public:
			Tracer *agtrc;
			Generic *agnet, *agnet2;
			floatt eloss, eloss2;
			sytet exec1, exec2;
			Flux *agloss, *agloss2, *agtrain, *agtrain2;

			NameScope *algol(Tracer *trc, Flux *ingate, Flux *targate, intt latent_sz,
				intt indiscret, intt outdiscret, intt embedim, bool contraction, sytet af, 
				floatt lr, const bytet *name);
			Algol(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				bool contraction = 1, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "algol");
			Algol(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				bool contraction = 1, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "algol");
			void recording(void) {}
			Flux *train(intt *n_train = 0);
			Flux *predict(Flux **loss_fx = 0);
			Flux *loss2(void);
			void accuracy(Flux *&predicts, Flux *&targets, intt discrete_out = -1);
			Flux *measureAccuracy(void);
		};
		Algol *algol(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			bool contraction, sytet af, floatt lr, const bytet *name);

		class Stratus : public Cell {
		public:
			Tracer *sortrc, *tartrc;
			Generic *srsor_net, *srnet3, *srtar_net, *srrtar_net;
			Flux *snet_ingate, *snet_targate;
			Flux *srsor_loss, *srtar_loss, *srrtar_loss, *srsor_train, *tarout;
			intt ichkBound, ichkBound2;
			bool endChkBound, endChkBound2;

			NameScope *stratus(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name);
			Stratus(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "stratus");
			Stratus(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "stratus");
			void recording(void) {}
			Flux *train(intt *n_train = 0);
			Flux *predict(Flux **loss_fx = 0);
			Flux *loss2(void);
			void accuracy(Flux *&predicts, Flux *&targets, intt discrete_out = -1);
			Flux *measureAccuracy(void);
		};
		Stratus *stratus(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af, floatt lr, const bytet *name);

		void BoostMemput(intt gid = 0, sytet boot_if = 0);

#define ERR_MSG_HEAD	"Error Code: %d "
		class FaultObj {
		public:
			intt fltcd;
			intt fltoff;
			bytet fltmsg[1024];
		};
		void throwFault(intt tflt, const char *fmt, ...);
		void printo(vector<Flux *> *fxl);
		void printg(vector<Flux *> *fxl);
		void dumpToGrad(vector<Flux *> *fxl);
		void resetGrad(vector<Flux *> *fxl);
		Flux *concat(vector<Flux *> *fxl, intt axis);
		Flux *concat(initializer_list<Flux *> fxl, intt axis);
		Flux *stack(vector<Flux *> *fxl, intt axis);
		Flux *stack(initializer_list<Flux *> fxl, intt axis);
		void lbackward(vector<Flux *> *fxl, void *tcxt = nullptr);
		void *mp_pointer_t(void);
	}
}