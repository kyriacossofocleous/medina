# 1 "messy_mecca_kpp_acc.cu"
# 3842 "messy_mecca_kpp_acc.cu"
double *temp_gpu = 0;
double *press_gpu = 0;
double *cair_gpu = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_conc = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_temp = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_press = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_cair = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_khet_st = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_khet_tr = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_jx = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_jac0 = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_Ghimj = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_varNew = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_K = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_varErr = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_dFdT = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_Fcn0 = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_var = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_fix = 0;
# 4326 "messy_mecca_kpp_acc.cu"
double *d_rconst = 0;
extern int initialized;


double *d_rstatus = 0;
# 4330 "messy_mecca_kpp_acc.cu"
double *d_absTol = 0;
# 4330 "messy_mecca_kpp_acc.cu"
double *d_relTol = 0;
int *d_istatus = 0;
# 4331 "messy_mecca_kpp_acc.cu"
int *d_istatus_rd = 0;
# 4331 "messy_mecca_kpp_acc.cu"
int *d_xNacc = 0;
# 4331 "messy_mecca_kpp_acc.cu"
int *d_xNrej = 0;
struct int4 *d_tmp_out_1 = 0;
# 4332 "messy_mecca_kpp_acc.cu"
struct int4 *d_tmp_out_2 = 0;
# 4737 "messy_mecca_kpp_acc.cu"
double conc[426240];
double temp[5760];
double press[5760];
double cair[5760];
double jx[426240];


int xNacc[5760];
int xNrej[5760];

extern double conc_cell[74];
# 4824 "messy_mecca_kpp_acc.cu"
extern double abstol[74];



extern double reltol[74];




extern double khet_st[426240];



extern double khet_tr[426240];
# 4327 "messy_mecca_kpp_acc.cu"
int initialized = 0;
# 4747 "messy_mecca_kpp_acc.cu"
double conc_cell[74] = {(0.0),(0.0),(1.130030837133365053e-06),(2161.176818259260017),(0.0001469481417859824128),(0.0002894067546497780248),(0.0),(0.0),(6.377486492629031622e-31),(0.0002774602114035594155),(9.159068418074057955e-22),(1.681545841334170886e-30),(6.587848965925120834e-36),(4.057130203198297654e-31),(7.556675262619906408e-06),(5.625822089563362005e-06),(7.248546508346979966e-10),(7.771754415762507499e-39),(1.672965892516880913e-32),(5.778276640099592545e-29),(2.169623196599309996e-31),(4.449685524913890094e-29),(9.236991853178720775e-28),(1.73125484793541291e-09),(6.419363370200839028e-28),(4.035724058634079279e-29),(6234.08726448301968),(25802.77881328489821),(1.33974252411334005),(11.15141769464590027),(8.023966161170008037e-32),(1.405402576145367075e-30),(2.416365419045455886e-29),(3.763980220765518692e-33),(0.0003687747273615521091),(4.4006958058575549e-30),(8.096351349854846844e-09),(1.605777396541510016e-08),(8.424266813161654464e-05),(1.275728897910597132e-29),(36780.60690670069744),(44.28021855848810162),(5.485594561042763652e-10),(3.418234885986840192e-32),(1.808885697309332159e-08),(2.295321288609201868e-30),(7.186736555958002736e-32),(667193926.5490679741),(9.443976722997097911e-30),(2.065479750965849984e-30),(658798139.7173529863),(5013220.829272099771),(6.594652607797343386e-13),(4.779051920325237223e-33),(0.2413303920517579915),(2.657031589287186106e-30),(1.166890334972386016e-14),(337.0697822316579959),(126494.9772056910006),(891.1969152016109774),(222.5573672438320045),(1.224516246698130084),(4845.027548231059882),(535329.6161963680061),(0.03077774956209535992),(989833722.9372060299),(38527.62914324420126),(1.857293910861109019e-07),(5035616002.440179825),(26824247.31079050153),(211466.2391751630057),(60638129767802.70312),(225227339137553.0),(87651408241.11650085)};
# 4824 "messy_mecca_kpp_acc.cu"
double abstol[74] = {(0.0)};



double reltol[74] = {(0.0)};




double khet_st[426240] = {(0.0)};



double khet_tr[426240] = {(0.0)};
