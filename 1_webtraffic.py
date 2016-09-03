import scipy as sp
data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
print(data[:10])

#we separate the dimensions in two vectors
x=data[:,0]
y=data[:,1]

#we check how many missing values there are
sp.sum(sp.isnan(y))

#there are 8 missing values. we can afford to remove them. that's what we do next
x=x[~sp.isnan(y)]
y=y[~sp.isnan(y)]

#let's visualize the data
import matplotlib.pyplot as plt
plt.scatter(x,y,s=10)		#dots of size 10
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
			['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
#draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-',color='0.75')
plt.show()

#now we will look for the right model behind the data points and a learning algorithm
def error(f,x,y):
	return sp.sum((f(x)-y)**2)
	
#starting with a simple straight line
fp1, residuals, rank, sv, rcond=sp.polyfit(x,y,1,full=True)
print("Model parameters: %s" % fp1)
print(residuals)

#we create a model function from the model parameters
f1=sp.poly1d(fp1)
print(error(f1,x,y))

#we plot our first trained model
plt.scatter(x,y,s=10)		#dots of size 10
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
			['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
#draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-',color='0.75')

fx=sp.linspace(0,x[-1],1000)	#generate x-values for plotting
plt.plot(fx,f1(fx),linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")
plt.show()

#polynomial of degree 2
f2p=sp.polyfit(x,y,2)
print(f2p)
f2=sp.poly1d(f2p)
print(error(f2,x,y))


plt.scatter(x,y,s=10)		#dots of size 10
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
			['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
#draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-',color='0.75')

fx=sp.linspace(0,x[-1],1000)	#generate x-values for plotting
plt.plot(fx,f1(fx),linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")
plt.plot(fx,f2(fx),linewidth=4)
plt.legend(["d=%i" % f2.order], loc="upper left")
plt.show()

#it seems there is an inflection point between weeks 3 and 4. so let's separate the data and train two lines using week 3.5 as inflection point
inflection=3.5*7*24		#calculate the inflection point in hours
xa=x[:inflection]	#data before the inflection point
ya=y[:inflection]
xb=x[inflection:]
yb=y[inflection:]

fa=sp.poly1d(sp.polyfit(xa,ya,1))
fb=sp.poly1d(sp.polyfit(xb,yb,1))

fa_error=error(fa,xa,ya)
fb_error=error(fb,xb,yb)
print("Error inflection=%f" %(fa_error+fb_error))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#in the book, they show that the polynomial of degree 2 is the one that fits the data the best.
#now, let's try to answer the initial question
fbt2=sp.poly1d(sp.polyfit(X_train,y_train,2))
print("fbt2(x)=\n%s" % fbt2)
print("fbt2(x)-100,000= \n%s" % (fbt2-100000))		#recall we want to know when we will reach 100000 requests per hour

from scipy.optimize import fsolve	#fsolve finds the root of the polynomial
reached_max=fsolve(fbt2-100000,x0=800)/(7*24)	#we need to provide a starting position: 800 because we have 743 entries and each of them represents 1 hour
print("100,000 hits per hour expected at week %f" % reached_max[0])













