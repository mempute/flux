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

		class Tracer {
		public:
			bytet characterName[24];
			intt characterType, tcr_reserve;
			virtual ~Tracer() {}
			virtual void namescope(bytet *nsm) {}
			virtual void endscope(void) {}
			virtual void init_train(void) = 0;
			virtual void run(Flux *target) = 0;
			virtual void run(vector<Flux *> target) = 0;
			virtual void run(vector<Flux *> *target) = 0;
			virtual void npset(intt n) = 0;
			virtual void lapset(sytet lap) = 0;
			virtual void gprset(floatt d) = 0;
			virtual void modeset(sytet cpm) = 0;
			virtual void portingGraph(Tracer *target) = 0;
			virtual Flux *getFlux(Flux *sfx) = 0;
			virtual void promptMode(bool on) = 0;
			virtual void sizeBatch(intt sz) = 0;
			virtual void saveWeight(void) = 0;
			virtual intt loadWeight(void) = 0;
			virtual void truncWeight(void) = 0;
			virtual void printWeight(void) = 0;
			virtual void traceopt(intt i, doublet v) = 0;
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

		class Generic {
		public:
			void *operator new(size_t size, Tracer *tcr);
		};

		class Contact : public Generic {
		public:
			sytet Tgener = GEN_T_TACT;
			Contact *ptrLeft, *ptrRight;
			Generic *vcontact;
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
#define ACTF2_PRELU	100
		typedef intt(*vinitfp)(Flux *);

		class Flux : public Generic {
		public:
			sytet Tgener = GEN_T_FLUX;
			ubytet qType, fxType;
			intt nRefer, ibRefer, nbRefer;
			Generic *bwAp, *bwLink, *directLink;
			Contact *fwAp;
			void *quantum;
			intt fdim, fxSize;
			intt fshape[MX_DIM];
			bytet *fxName;
			Tracer *fxTcr;
			intt backwv;
			//intt nJoint;
			intt bregid;
			bool termifx;
			bool meanAfter;
			bool trainherit;
			vinitfp vinitter;
			void *cursorP;
			Flux *ptrLeft, *ptrRight;//lstTarget사용

			void init(Tracer *tcr, ubytet qtype, ubytet fxtype, vinitfp vfp, const bytet *name);
			intt sizef(void);
			intt sizem(bool t_size = 0);
			Flux(Tracer *tcr, intt *axid, intt ndim, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
			Flux(Tracer *tcr, initializer_list<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
			Flux(Tracer *tcr, vector<intt> &axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);

			void instTens(bool inst);
			void fdims(intt ndim, intt *pdim);
			void fdims(initializer_list<intt> axid);
			bool checkInBwfx(bool lock);
			bool checkInBwfx2(void);
			void projected(Generic *ap);
			void referenced(Generic *ap);
			void exec(void *tcxt);
			void backward(void);
			void backend_nstep(Flux *fxp, Flux *fxs, Flux *fxo, void *ap);
			void backend_ps(Flux *fxp, Flux *fxs, Flux *fxo, void *ap);
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
			void resizing2(intt n);
			void resizing2(Flux *src);
			void sizeCheck(intt ndim, intt axid[]);
			void resizing(Flux *src);
			Flux *reshape(vector<intt> axid);
			Flux *bypass(const bytet *msg = nullptr);
			Flux *adjust(Flux *in);
			Flux *expand_dims(intt axis);
			Flux *squeeze(intt axis = -1);
			Flux *transpose(vector<intt> axid);
			Flux *softmax(void);
			Flux *squaredDifference(Flux *fxs);
			Flux *softmaxCrossEntropy(Flux *fxt);
			Flux *sum(void);
			Flux *mean(void);
			Flux *meanSquareError(Flux *fxt);
			Flux *actf(intt actf_op);
			Flux *tanh(void);
			Flux *relu(void);
			Flux *sigmoid(void);
			Flux *prelu(floatt iv);
			Flux *sqrt(void);
			Flux *log(void);
			Flux *embedding_lookup(Flux *fxi);
			Flux *one_hot(intt depth, doublet on_value = 1, doublet off_value = 0, intt axis = -1, intt dtype = -1);
			void boundSlice(intt n, intt code[], intt idx[], bool check);
			Flux *slice(vector<vector<intt>> axis);
			Flux *argmax(intt axis, sytet t_out = -1);
			Flux *equal(Flux *fxs, doublet cmpv, bool cscalr, bool eq);
			Flux *clipValue(doublet low, doublet high);
			intt copyt(void *psrc, ubytet tsrc, intt bsz);
			void feedt(void *psrc, ubytet tsrc, intt sz);
			intt copyf(void *pdat, intt bsz);
			intt copyf(Flux *src);
			void feedf(void *pdat, intt sz);
			void feedf(Flux *src);
			Flux *feedf(ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
			void dstrw(const bytet dstr[]);
			void dumpToGrad(void);
			void resetGrad(void);
			void shape(void);
			void *begin_p(void);
			void *begin_wp(void);
			void *end_p(void);
			doublet at_d(intt i);
			doublet at_d2(intt i);
			void printo(bool nwl = true, sytet leaf_one = 0);
			void printg(bool nwl = true, sytet leaf_one = 0);
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
			Flux *layer_dense(intt nout, intt actf_code, vinitfp vfp = Initializer::xavier, const bytet *name = nullptr);
			Flux *layer_dense(intt nout, const bytet *actf, vinitfp vfp = Initializer::xavier, const bytet *name = nullptr);
			Flux *layer_normal(void);
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
			Flux *coupledot(Flux *wsizing, Flux *wb);
			Flux *trippledot(Flux *wsizing, Flux *wb);
			Flux *stridedot(Flux *wsizing, Flux *wb);
		};

		const ubytet variable = 0, trainable = 1, apply = 2, constant = 3, const_apply = 4;
		const ubytet tshort = 2, tfloat = 3, tint = 4, tlong = 5, tdouble = 6;

		Flux *flux(Tracer *tcr, initializer_list<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Tracer *tcr, vector<intt> axid, ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Tracer *tcr, Flux *src, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Flux *src, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Tracer *tcr, const bytet dstr[], ubytet qtype=tfloat, ubytet fxtype = constant, vinitfp vfp = nullptr, const bytet *name = nullptr);
		Flux *flux(Tracer *tcr, intt ndim, intt axid[], ubytet qtype, ubytet fxtype, vinitfp vfp = nullptr, const bytet *name = nullptr);

		class Optimizer : public Generic {
		public:
			intt bregid;
			floatt rLning;
			Optimizer(floatt lr)
			{
				rLning = lr;
			}
			virtual Flux *minimize(Flux *fx) = 0;
		};
		class AdamOptimizier : public Optimizer {
		public:
			AdamOptimizier(floatt lr) : Optimizer(lr)
			{
			}
			Flux *minimize(Flux *fx);
		};
		AdamOptimizier *adam_optimizer(Tracer *tcr, floatt lr = -1);

		class GradientDescentOptimizier : public Optimizer {
		public:
			GradientDescentOptimizier(floatt lr) : Optimizer(lr)
			{
			}
			Flux *minimize(Flux *fx);
		};
		GradientDescentOptimizier *gradient_descent_optimizer(Tracer *tcr, floatt lr = -1);

		class Coaxial : public Generic {//c language neural network
		public:
			void *canet;
			Flux *zcodec, *dcodec, *clogit, *cypred, *closs, *coptrain, *caccuracy;
			Flux *ctargets, *cpredictions, *copmeasure;
			void cbuild(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
			intt cbuild(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
			void coaxial(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim, sytet af, floatt lr, const bytet *name);
			Coaxial(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
			Coaxial(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
			Coaxial(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
			Coaxial(Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
				sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
			void accuracy(Flux *predicts, Flux *targets);
			Flux *measureAccuracy(void);
			intt reduction(intt outsz, intt szlat = -1);
			void decompose(intt outsz);
			intt decompose2(intt outsz, intt szlat = -1);
			void connect(Flux *targate, intt outdiscret, floatt lr = -1, sytet opt_t = 0);
			Flux *train(void);
			Flux *predict(Flux **loss_fx = 0);
		};
		Coaxial *coaxial(Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
		Coaxial *stepwise(Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);
		Coaxial *coaxial(Tracer *tcr, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet");
		Coaxial *stepwise(Tracer *tcr, Flux *ingate, intt outsz, intt latent_sz, intt indiscret, intt embedim,
			sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = "cnet", bool auto_encoder = 0);

		void BoostMemput(void);

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