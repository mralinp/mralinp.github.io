---
layout: post
title:  "Snake game"
author: "Ali N. Parizi"
img:    "/assets/images/snake-game/snake-game.jpeg"
date:   2022-03-21  18:15:32 +0330
categories: project game python entry-level
---

# 1. Intro
Hello there, this is Ali speaking. When i started studying computer engineering at Shiraz university, CSE101 was the very primarly course that we should take to get familiar with the basic concepts of programming. They introduced **python** to us for the first time. Python is a very cool programming language that have very waste practical areas in almost any majors such as Data science, machine learning, web-development, game development, etc.

Now i'm using python on almost my every day of my academic career. So it was a good choice to begin programming and get familiar with the concepts of programming. On the other hand, in the first year of high-school i learned **C** programming as my first language. I was using C to solve mathematic problems and only for fun because I didn't have any other source to learn, except our computer teacher and a very old resource which was based on **Borland C++** and was running on **MS-DOS**. That days we didn't have the access to the internet on our home or school so the only way of learning things was reading books and asking questions. So to me the course CSE101 was the easiest course of my life. I never studied anything about the course silabuses but I was just learning everything possible about python that could found. **Our final project was making a very simple snake game (like old school Nokia phones game) without using advanced libraries such as pyGame or other alternatives.**

## 1.1 Problem statement:

Develop a straightforward snake game such as the one on old-school Nokia phones. A single snake moves around the game world. Users can control the snake using arrows keys or W, A, S, and D to move UP, LEFT, DOWN, and RIGHT.

After a random time, it will be deployed an apple on a random spot on the map. If the snake eats the apple, its tail extends, and the player receives 100 points. The map can contain some walls or obstacles. If the snake hits a block or its tail, the game will be finished, and the player will lose the game.

By pressing the Esc key, the game pauses. You have to make the ability to store the highest score after the game finishes.

Any creative ideas and implementation that improve performance, game experience, and game appearance considers as bonuses.

# 2. Solution
Did you read **[part 1.1](#11-problem-statement)** carefully? As it could be found from the problem statement. We can break the problem into some parts and levels. First is to implement a moving object on the 2D game board which actually is our snake. As the game world is console and no graphical libraries are allowed here we need to think and consider about a simple world of char games. In this world, all objects in the game are nothing but characters. For example snake head can be displayed as the character '@', its tail segments can be displayed with 'o' and Apples are displayed with 'A'. At the end walls could be represented as '#'. First, we try to implement the basics of the game. The challenge here is how to print a character on a specific position of the screen. 

# 3. Implementation

# 4. Final words