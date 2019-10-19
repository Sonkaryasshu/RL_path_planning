# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, InstructionGroup
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []
dist_list = []
dist = 0
# Initializing the map
first_update = True
x_max = 800
y_max = 600
goal_x = 30
goal_y = 600 - 30
start_x =  x_max - goal_x
start_y = y_max - goal_y
sand = np.zeros((x_max,y_max))

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>x_max-10 or self.sensor1_x<10 or self.sensor1_y>y_max-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>x_max-10 or self.sensor2_x<10 or self.sensor2_y>y_max-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>x_max-10 or self.sensor3_x<10 or self.sensor3_y>y_max-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Goal(Widget): 
    pass
# Creating the game class

class Game(Widget):
    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        print("Game init called")
        self.ig = InstructionGroup()
        with self.canvas:
            Color(1,0,1)
        self.car.x = start_x
        self.car.y = start_y
        self.line = Line(points = (start_x, start_y), width = 1)
        self.ig.add(self.line)
        self.canvas.add(self.ig)
    
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global start_x
        global start_y
        global dist
        global dist_list
        global first_update
        
        if first_update:
            self.car.x = start_x
            self.car.y = start_y
            first_update = False
        self.line.points += [self.car.center_x, self.car.center_y]
        dist = dist + 1
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        self.goal.center = Vector(goal_x, goal_y)
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 20:
            self.car.x = start_x
            self.car.y = start_y
            self.line.points = [self.car.x, self.car.y]
            dist_list.append(dist)
            dist = 0
            # plt.plot(dist_list)
            # plt.title("Car")
            # plt.xlabel("No of iterations")
            # plt.ylabel("No of steps taken")
            # plt.show()
        last_distance = distance
    
# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    def build(self):
        self.parent = Game()
        self.parent.serve_car()
        # Clock.schedule_interval(self.parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear', size=(75,50), background_color = (1,0,0,0.5))
        savebtn = Button(text = 'save', pos = (75, 0), size=(75,50), background_color = (1,0,0,0.5))
        loadbtn = Button(text = 'load', pos = (2 * 75, 0), size=(75,50), background_color = (1,0,0,0.5))
        self.startbtn = Button(text = 'start', pos = (3 * 75, 0), size=(75,50), background_color = (1,0,0,0.5))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        self.startbtn.bind(on_release = self.start)
        self.parent.add_widget(self.painter)
        self.parent.add_widget(clearbtn)
        self.parent.add_widget(savebtn)
        self.parent.add_widget(loadbtn)
        self.parent.add_widget(self.startbtn)
        self.parent.car.center = (start_x, start_y)
        self.parent.ball3.pos = (Vector(30, 0).rotate(self.parent.car.angle) + self.parent.ball3.pos)
        self.parent.ball2.pos = (Vector(30, 0).rotate((self.parent.car.angle+30)%360) + self.parent.ball2.pos)
        self.parent.ball1.pos = (Vector(30, 0).rotate((self.parent.car.angle-30)%360) + self.parent.ball1.pos)
        return self.parent

    def start(self,obj):
        Clock.schedule_interval(self.parent.update, 1.0/60.0)
        self.startbtn.text=""

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((x_max,y_max))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
