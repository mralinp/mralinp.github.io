---
layout: post
title:  "Snake game"
author: "Ali N. Parizi"
img:    "/assets/images/snake-game/snake-game.jpeg"
date:   2022-03-21  18:15:32 +0330
categories: project game python entry-level
brief: "Snake game is a very cool and informative mini project for beginners, in this post we are going to implement a simple snake game from scratch using python."
---

# 1. Intro
Hello there, this is Ali speaking. When I started studying computer engineering at Shiraz University, CSE101 was the introductory course we should take to familiarize ourselves with basic programming concepts. There, I have introduced to the **python** programming language for the first time. Python is a very cool programming language that wastes practical areas in almost any major, such as data science, machine learning, web development, game development, etc.

<br>
<div align="center">
    <img src="/assets/images/snake-game/python-logo.png" width="200px">
</div>
<br>

Now, I'm using python almost every day of my academic career. So it was an excellent choice to begin programming and get familiar with programming concepts. On the other hand, in the first year of high school, I learned **C** programming as my first programming language. I was using C to solve mathematic problems and only for fun because I had no other source to learn except our computer teacher and an ancient resource based on **Borland C++** and running on **MS-DOS**. Those days we didn't have access to the internet in our home or school, so the only way of learning things was by reading books and asking questions. To me, CSE101 was the most straightforward course of my life. I never studied anything about the course syllabuses; I just learned everything possible about python that I could find on the internet. The course's final project was **"Making a Very Simple Snake Game"** (like old school Nokia phones game) without using advanced libraries or frameworks such as PyGame or other alternatives.

In this article, we are about to implement a **Very Simple Snake Game**. The problem statement is available in the next section.

# Table Of Contents
- [1. Intro](#1-intro)
- [2. Problem Statement](#2-problem-statement)
- [3. Challenges of The Project](#3-Challenges of The Project)
- [4. Solution and Design](#4.solution-and-design)
- [5. Implementation](#5-implementation)
- [6. Final words](#6-final-words)

# 2. Problem Statement:

Develop a straightforward snake game such as the one on old-school Nokia phones. A single snake moves around the game world. You can control the snake using arrow keys or `W`, `A`, `S`, and `D` to move `UP`, `LEFT`, `DOWN`, and `RIGHT`. Meanwhile, at any random time, it could be deployed an apple (or food) on a random spot on the map. If the snake eats the food, its tail extends by a unit, and the player receives 100 points. The map can contain some walls or obstacles. The player loses the game if the snake hits a block or its tail. Pressing the `Esc` key pauses and unpauses the game. You have to make the ability to store the achieved score after each game.
> Note: Any creative ideas and implementations that improve performance, game experience, and appearance will be considered as bonuses.

# 2. Solution
Did you read the **[problem statement](#11-problem-statement)** section carefully? As it could be found from the problem statement. We can break the problem into some parts and levels. First is to implement a moving object on the 2D game board which actually is our snake. As the game world is console and no graphical libraries are allowed here we need to think and consider about a simple world of char games. In this world, all objects in the game are nothing but characters. For example snake head can be displayed as the character `@`, its tail segments can be displayed with `o` and Apples are displayed with `A`. At the end walls could be represented as `#`. First, we try to implement the basics of the game. The challenge here is how to print a character on a specific position of the screen. 

# 3. Challenges of The Project

# 4. Solution and Design

# 5. Implementation

# 6. Final words