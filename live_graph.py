import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#ax2 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("sampleText.txt","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    yar2 = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y = eachLine.split(',')
            xar.append(float(x))
            yar.append(float(y))
            yar2.append(0.85)
    ax1.clear()
    ax1.plot(xar,yar)
    ax1.plot(xar,yar2)
#ani = animation.FuncAnimation(fig, animate, interval=1000)
animate(1)
plt.title('Mouth Aspect Ratio Graph')
plt.xlabel('time (s)')
plt.ylabel('Mouth Aspect Ration')
plt.show()
