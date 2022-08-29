import numpy as np
import matplotlib.pyplot as plt

def rate(coords, a, b, r, I):
    v = coords[0]
    w = coords[1]
    v_dot = v*(a - v)*(v - 1) - w + I
    w_dot = b*v - r*w
    return np.array([v_dot, w_dot])

def phase_plot(Iext, startPoints, figNum, imgName, annotation):
    a = 0.5
    b = 0.1
    r = 0.1
    I = Iext

    x = np.arange(-5, 4, 0.01)
    v = np.arange(-1.5, 2, 0.01)
    w = np.arange(-1.5, 2, 0.01)

    V, W = np.meshgrid(v, w)
    DV, DW = rate([V, W], a, b, r, I)

    v_null = x*(a-x)*(x-1) + I
    w_null = x

    plt.figure(figNum, figsize=((6, 6)))
    plt.plot(x, v_null, color="purple")
    plt.plot(x, w_null, color="red")
    plt.streamplot(V, W, DV, DW, density=2.0, color="darkgrey")
    if(startPoints != None):
        plt.streamplot(V, W, DV, DW, start_points=startPoints, density=2.0, color="blue")
    if(annotation != None):
        plt.annotate(annotation, annotation, textcoords="offset points", xytext=(10, 11), size=12)
    plt.legend(["v-nullcline", "w-nullcline"])
    plt.xlabel("V")
    plt.ylabel("w")
    plt.title("Phase plot")
    ax = plt.gca()
    ax.set_xlim([-1.5, 2])
    ax.set_ylim([-1.5, 2])
    plt.savefig(imgName)

def time_plot(Iext, v_init, figNum_V, figNum_W, imgName_V, imgName_W):
    a = 0.5
    b = 0.1
    r = 0.1
    I = Iext

    v_0 = v_init
    w_0 = 0

    delT = 0.0005
    t = np.array(np.arange(1000))

    v = np.array([v_0])
    w = np.array([w_0])

    for i in t:
        if i == 0:
            continue 
        v_prev = v[i - 1]
        w_prev = w[i - 1]

        v = np.append(v, v_prev + ((v_prev*(a-v_prev)*(v_prev - 1)) - w_prev + I)*(i*delT))
        w = np.append(w, w_prev + (b*v_prev - r*w_prev)*i*delT)

    print(len(v), len(t))
    plt.figure(figNum_V, figsize=((6, 6)))
    plt.plot(t, v)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Voltage vs Time\nIext = {0}; V(0) = {1}; w(0) = 0".format(Iext, v_0))
    plt.savefig(imgName_V)

    plt.figure(figNum_W, figsize=((6, 6)))
    plt.plot(t, w)
    plt.xlabel("Time (ms)")
    plt.ylabel("W(t)")
    plt.title("W(t) vs Time\nIext = {0}; V(0) = {1}; w(0) = 0".format(Iext, v_0))
    plt.savefig(imgName_W)



#########################################################################################
### Case 1(a): I_ext = 0; 
### Phase plot of V-nullcline and W-nullcline
#########################################################################################

I = 0
phase_plot(I, None, 1, "Q1_a", None)


#########################################################################################
### Case 1(b)(i): I_ext = 0; 
### V(0) < a; V(0) = 0.4; w(0) = 0;
#########################################################################################

I = 0
time_plot(0, 0.4, 2, 3, "Q1_b_i_1", "Q1_b_i_2"  )
phase_plot(I, [[0.4, 0], [0, 0]], 4, "Q1_b_i_3", (0.4, 0))

#########################################################################################
### Case 1(b)(ii): I_ext = 0; 
### V(0) > a; V(0) = 0.6; w(0) = 0;
#########################################################################################

I = 0
time_plot(0, 0.6, 5, 6, "Q1_b_ii_1", "Q1_b_ii_2"  )
phase_plot(I, [[0.6, 0], [0, 0]], 7, "Q1_b_ii_3", (0.6, 0))

#########################################################################################
### Case 2; Iext = 0.23
### Finding I1
#########################################################################################

I = 0.23 
time_plot(I, 0.4, 8, 9, "Q2_1", "Q2_2"  )
time_plot(I, 0.6, 10, 11, "Q2_3", "Q2_4"  )


#########################################################################################
### Case 2; Iext = 0.80
### Finding I2
#########################################################################################

I = 0.80
time_plot(I, 0.4, 12, 13, "Q2_5", "Q2_6"  )
time_plot(I, 0.6, 14, 15, "Q2_7", "Q2_8"  )


#########################################################################################
### Case 2(a); Iext = 0.5
### Phase plot at Iext = 0.5
#########################################################################################

I = 0.5
phase_plot(I, None, 16, "Q2_a", None)


#########################################################################################
### Case 2(b); Iext = 0.5
### V(0) < a; V(0) = 0.4; w(0) = 0;
#########################################################################################

I = 0.5
phase_plot(I, [[0.4, 0],[0.5, 0.5], [1.5, 1.5], [-1.5, 1.5]], 17, "Q2_b_1", None)

#########################################################################################
### Case 2(c); Iext = 0.5
### V(0) < a; V(0) = 0.4; w(0) = 0;
#########################################################################################

I = 0.5
time_plot(I, 0.4, 19, 20, "Q2_c_1", "Q2_c_2"  )

#########################################################################################
### Case 2(c); Iext = 0.5
### V(0) > a; V(0) = 0.6; w(0) = 0;
#########################################################################################

I = 0.5
time_plot(I, 0.6, 21, 22, "Q2_c_3", "Q2_c_4"  )

#########################################################################################
### Case 3(a); Iext = 1.0
### Phase plot at Iext = 1.0
#########################################################################################

I = 1.0
phase_plot(I, None, 23, "Q3_a", None)

#########################################################################################
### Case 3(b); Iext = 1.0
### Phase plot at Iext =1.0
#########################################################################################

I = 1.0
phase_plot(I, [[0.4, 0]], 24, "Q3_b", (0.4, 0))

#########################################################################################
### Case 3(c); Iext = 1.0
### V(0) < a; V(0) = 0.4; w(0) = 0;
#########################################################################################

I = 1
time_plot(I, 0.4, 25, 26, "Q3_c_1", "Q3_c_2"  )



#########################################################################################
### Case 3(c); Iext = 1.0
### V(0) > a; V(0) = 0.6; w(0) = 0;
#########################################################################################

I = 1
time_plot(I, 0.6, 27, 28, "Q3_c_3", "Q3_c_4"  )

#########################################################################################
### Case 4(a); Iext = 0; b/r = 0;
### Bistability Phase Plot
#########################################################################################

a = 0.5
b = 0
r = 1
I = 0

x = np.arange(-5, 4, 0.01)
v = np.arange(-1.5, 2, 0.01)
w = np.arange(-1.5, 2, 0.01)

V, W = np.meshgrid(v, w)
DV, DW = rate([V, W], a, b, r, I)

v_null = x*(a-x)*(x-1) + I
w_null = (b/r)*x

plt.figure(29, figsize=((6, 6)))
plt.plot(x, v_null, color="purple")
plt.plot(x, w_null, color="red")
plt.streamplot(V, W, DV, DW, density=2.0, color="darkgrey")
plt.streamplot(V, W, DV, DW, start_points=[[-1, 0.5]], density=2.0, color="blue")
plt.streamplot(V, W, DV, DW, start_points=[[0.6, 0]], density=2.0, color="green")
plt.streamplot(V, W, DV, DW, start_points=[[1.25, 0.5]], density=2.0, color="magenta")
# plt.annotate(annotation, annotation, textcoords="offset points", xytext=(10, 11), size=12)
plt.legend(["v-nullcline", "w-nullcline"])
plt.xlabel("V")
plt.ylabel("w")
plt.title("Phase plot")
ax = plt.gca()
ax.set_xlim([-1.5, 2])
ax.set_ylim([-1.5, 2])
plt.savefig("Q4_a")

#########################################################################################
### Case 4(c); Iext = 0; b/r = 0;
### V(0) < a; V(0) = 0.4; W(0) = 0;
#########################################################################################

a = 0.5
b = 0
r = 1
I = 0

v_0 = 0.4
w_0 = 0

delT = 0.0005
t = np.array(np.arange(1000))

v = np.array([v_0])
w = np.array([w_0])

for i in t:
   if i == 0:
       continue 
   v_prev = v[i - 1]
   w_prev = w[i - 1]

   v = np.append(v, v_prev + ((v_prev*(a-v_prev)*(v_prev - 1)) - w_prev + I)*(i*delT))
   w = np.append(w, w_prev + (b*v_prev - r*w_prev)*i*delT)

plt.figure(30, figsize=((6, 6)))
plt.plot(t, v)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage vs Time\nIext = {0}; V(0) = {1}; w(0) = 0".format(I, v_0))
plt.savefig("Q4_c_1")

plt.figure(31, figsize=((6, 6)))
plt.plot(t, w)
plt.xlabel("Time (ms)")
plt.ylabel("W(t)")
plt.title("W(t) vs Time\nIext = {0}; V(0) = {1}; w(0) = 0".format(I, v_0))
plt.savefig("Q4_c_2")

#########################################################################################
### Case 4(c); Iext = 0; b/r = 0;
### V(0) > a; V(0) = 0.6; W(0) = 0;
#########################################################################################

a = 0.5
b = 0
r = 1
I = 0

v_0 = 0.6
w_0 = 0

delT = 0.0005
t = np.array(np.arange(1000))

v = np.array([v_0])
w = np.array([w_0])

for i in t:
   if i == 0:
       continue 
   v_prev = v[i - 1]
   w_prev = w[i - 1]

   v = np.append(v, v_prev + ((v_prev*(a-v_prev)*(v_prev - 1)) - w_prev + I)*(i*delT))
   w = np.append(w, w_prev + (b*v_prev - r*w_prev)*i*delT)

plt.figure(32, figsize=((6, 6)))
plt.plot(t, v)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage vs Time\nIext = {0}; V(0) = {1}; w(0) = 0".format(I, v_0))
plt.savefig("Q4_c_3")

plt.figure(33, figsize=((6, 6)))
plt.plot(t, w)
plt.xlabel("Time (ms)")
plt.ylabel("W(t)")
plt.title("W(t) vs Time\nIext = {0}; V(0) = {1}; w(0) = 0".format(I, v_0))
plt.savefig("Q4_c_4")


plt.show()
