function transpose(matrix) {
  return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

function Projection_RK4(y0,Paramaters,t0,tf,h ) {

  h = h*1;
  let Results = new Array();

  let n = parseInt((tf - t0) / 86400000 / h, 10)+1;
  let t = new Array();
  t[0] = new Date(t0);

  Results[0] = new Array();
  Results[0][0] = t[0];
  for(let i=0; i< y0.length; i++){
    Results[0][i+1] = y0[i]*1;
  }
  let k0 = Model(y0,Paramaters);
  for(let i=y0.length; i< k0.length+y0.length; i++){
    Results[0][i+1] = k0[i-y0.length]*1;
  }

  tt = t0*1;
  for(let i = 1; i <= n; i++){
    tt += 86400000*h;
    t[i] = new Date(tt);

    let k1 = Model(y0,Paramaters);
    let y1 = new Array();
    for (let i=0;i<k1.length;i++){
      y1[i] = y0[i]*1 + h*k1[i]/2;
    }

    let k2 = Model(y1,Paramaters);
    let y2 = new Array();
    for (let i=0;i<k2.length;i++){
      y2[i] = y0[i]*1 + h*k2[i]/2;
    }

    let k3 = Model(y2,Paramaters);
    let y3 = new Array();
    for (let i=0;i<k3.length;i++){
      y3[i] = y0[i]*1 + h*k3[i];
    }

    let k4 = Model(y3,Paramaters);
    let y4 = new Array();

    for (let i=0;i<k4.length;i++){
      y4[i] = y0[i]*1 + h*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])/6.0;
    }

    Results[i] = new Array();
    Results[i][0] = t[i];
    for(let j=0; j< y0.length; j++){
      Results[i][j+1] = y4[j];
    }
    let k5 = Model(y4,Paramaters);
    for(let j=y0.length; j< k0.length+y0.length; j++){
      Results[i][j+1] = 1*k5[j-y0.length]; // k0[j-y0.length]*1;
    }


    y0 = y4;

  }
  return Results;

}

function Projection_RK45(y0,Paramaters,t0,tf,tol,hmax,hmin) {

  let Results = new Array();
  let hh = new Array();
  let rr = new Array();

  Results[0] = "test";

  if (hmax < hmin){
    return "Error: hmax is less than hmin!";
  }
  if (tf < t0){
    return "Error: tf is less than t0!";
  }

  // Because the input of hmin and hmax is assumed to be in days we convert it to milliseconds
  hmin = hmin*1*86400000;
  hmax = hmax*1*86400000;

  h = hmax;
  dimension = y0.length;
  tf = tf*1;
  t = t0*1;


  // See https://stackoverflow.com/questions/65416794/issue-on-runge-kutta-fehlberg-algorithm for source
  a2  =   2.500000000000000e-01;  //  1/4
  a3  =   3.750000000000000e-01;  //  3/8
  a4  =   9.230769230769231e-01;  //  12/13
  a5  =   1.000000000000000e+00;  //  1
  a6  =   5.000000000000000e-01;  //  1/2

  b21 =   2.500000000000000e-01;  //  1/4
  b31 =   9.375000000000000e-02;  //  3/32
  b32 =   2.812500000000000e-01;  //  9/32
  b41 =   8.793809740555303e-01;  //  1932/2197
  b42 =  -3.277196176604461e+00;  // -7200/2197
  b43 =   3.320892125625853e+00;  //  7296/2197
  b51 =   2.032407407407407e+00;  //  439/216
  b52 =  -8.000000000000000e+00;  // -8
  b53 =   7.173489278752436e+00;  //  3680/513
  b54 =  -2.058966861598441e-01;  // -845/4104
  b61 =  -2.962962962962963e-01;  // -8/27
  b62 =   2.000000000000000e+00;  //  2
  b63 =  -1.381676413255361e+00;  // -3544/2565
  b64 =   4.529727095516569e-01;  //  1859/4104
  b65 =  -2.750000000000000e-01;  // -11/40

  r1  =   2.777777777777778e-03;  //  1/360
  r2  =   0;
  r3  =  -2.994152046783626e-02;  // -128/4275
  r4  =  -2.919989367357789e-02;  // -2197/75240
  r5  =   2.000000000000000e-02;  //  1/50
  r6  =   3.636363636363636e-02;  //  2/55

  c1  =   1.157407407407407e-01;  //  25/216
  c2  =   0;
  c3  =   5.489278752436647e-01;  //  1408/2565
  c4  =   5.353313840155945e-01;  //  2197/4104
  c5  =  -2.000000000000000e-01;  // -1/5


  // Fill first row of Results with t0, y0 and dy0dt0

  Results[0] = new Array();
  Results[0][0] = t0;
  
  for(let i=0; i< dimension; i++){
    Results[0][i+1] = y0[i]*1;
  }

  let k0 = Model(y0,Paramaters);

  for(let i=dimension; i< 2*dimension; i++){
    Results[0][i+1] = k0[i-dimension]*1;
    Results[0][i+dimension+1] = 0;
  }

  n = 0;
  nn = 0;
  while ( (t*1 < tf*1)) {
    nn += 1;
    hh[nn] = h;

    if (t + h > tf){
      h = tf - t;
    }

    let donk = Model(y0,Paramaters);
    let k1 = new Array();
    let y1 = new Array();
    for (let i=0;i<dimension;i++){
      k1[i] = h * donk[i] / 86400000;
      y1[i] = y0[i]*1 + b21*k1[i];
    }

    donk = Model(y1,Paramaters);
    let k2 = new Array();
    let y2 = new Array();
    for (let i=0;i<dimension;i++){
      k2[i] = h * donk[i] / 86400000;
      y2[i] = y0[i]*1 + b31*k1[i] + b32*k2[i];
    }

    donk = Model(y2,Paramaters);
    let k3 = new Array();
    let y3 = new Array();
    for (let i=0;i<dimension;i++){
      k3[i] = h * donk[i] / 86400000;
      y3[i] = y0[i]*1 + b41*k1[i] + b42*k2[i] + b43*k3[i];
    }

    donk = Model(y3,Paramaters);
    let k4 = new Array();
    let y4 = new Array();
    for (let i=0;i<dimension;i++){
      k4[i] = h * donk[i] / 86400000;
      y4[i] = y0[i]*1 + b51*k1[i] + b52*k2[i] + b53*k3[i] + b54*k4[i];
    }

    donk = Model(y4,Paramaters);
    let k5 = new Array();
    let y5 = new Array();
    for (let i=0;i<dimension;i++){
      k5[i] = h * donk[i] / 86400000;
      y5[i] = y0[i]*1 + b61*k1[i] + b62*k2[i] + b63*k3[i] + b64*k4[i] + b65*k5[i];
    }

    donk = Model(y5,Paramaters);
    let k6 = new Array();
    for (let i=0;i<dimension;i++){
      k6[i] = h * donk[i] / 86400000;
    }

    let r = new Array();
    for (let i=0;i<dimension;i++){
      r[i] = Math.abs((r1 * k1[i] + r2 * k2[i] + r3 * k3[i] + r4 * k4[i] + r5 * k5[i] + r6 * k6[i])/h);
    }

    let rmax = 0;
    for (let i=0;i<dimension;i++){
      if (r[i] > rmax){
        rmax = r[i];
      }
    }

    rr[nn] = rmax;

    if (rmax <= tol){
      t += h;
      n += 1;
      Results[n] = new Array();
      Results[n][0] = new Date(t);
      for (let i=0;i<dimension;i++){
        Results[n][i+1] = y0[i]*1 + c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i];
        if (Results[n][i+1] < 0){
          Results[n][i+1] == 0;
        }
        y0[i] = Results[n][i+1]
      }
      donk = Model(y0,Paramaters);
      for(let i=dimension; i< 2*dimension; i++){
        Results[n][i+1] = 1*donk[i-dimension] ;
        Results[n][i+dimension+1] = r[i-dimension];
      }
    } /*else {
      return (new Date(t));//Results;
    }*/
    
    /*if (nn > 10000){
      return rr;
    }*/

    h = h * Math.min(Math.max(Math.pow(0.84 * (tol / rmax ),0.25),0.1),4.0);
    if (h > hmax ){
      h = hmax;
    } else if (h < hmin){
      return "Error at "+(new Date(t))+": Need a stepsize of "+h/86400000+" days, or "+h +" milliseconds!!";
    }
  }

  //return h;
  return Results; //(new Date(t));

}

function Model(y,Paramaters) {
  let beta = Paramaters[0]*1.0;
  let gamma = Paramaters[1]*1.0;
  let epsilion = Paramaters[2]*1.0;
  let delta = Paramaters[3]*1.0;
  let alpha = Paramaters[4]*1.0;
  let phi = Paramaters[5]*1.0;
  let zeta = Paramaters[6]*1.0;
  let Lambda = Paramaters[7]*1.0;
  let eta = Paramaters[8]*1.0;
  let ICUProportion = Paramaters[9];
  let HospMax = Paramaters[10];
  let ICUMax = Paramaters[11];

  let N = y[0];
  let S = y[1];
  let E = y[2];
  let I = y[3];
  let R = y[4];
  let D = y[5];
  let H = y[6];

  let Z = H * ICUProportion;
  let Y = H - Z;

  let Hospital_Strain_Factor = 1;
  if(Z > ICUMax){
    Hospital_Strain_Factor *= ((Z / ICUMax) -1)*ICUProportion + 1;
  }

  if(Y > HospMax){
    Hospital_Strain_Factor *= ((Y / HospMax) -1)*(1-ICUProportion) + 1;
  }

  let dydt = new Array();
  // dydt[0] = 0;//Lambda - eta*N - Hospital_Strain_Factor*zeta*H;
  dydt[1] = Lambda - eta*S - beta*I*S/N + delta*R;
  dydt[2] = beta*I*S/N - (epsilion + eta)*E;
  dydt[3] = epsilion*E - (gamma + eta + alpha)*(I);
  dydt[4] = gamma*I - (delta + eta)*R + (phi-zeta)*H;

  dydt[5] = Hospital_Strain_Factor*zeta*H;

  dydt[6] = alpha*(I-H) - ((phi-zeta) - eta)*H - dydt[5];

  dydt[0] = Lambda - eta*(N-H) - dydt[5];

  return dydt;
}
