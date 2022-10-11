---
layout: post
title: "Variational AutoEncoders: An Introduction"
author: "Ali N. Parizi"
img: "/assets/images/vae/title.png"
date: 2022-10-11 8:24:05 +0330
categories: blog ai machine-learning deep-learning
brief: "Variational AutoEncoders are very popular extention of autoencoders which try to map the input data into a probablistic distribution of the data in latent space instead of learning a direc representation. This article is a straightforward walkthrough to get familiar with Variational AutoEncoders."
---

# 1. Intro
In machine learning, a variational autoencoder, is an artificial neural network architecture introduced by Diederik P. Kingma and Max Welling, belonging to the families of probabilistic graphical models and variational Bayesian methods. Variational autoencoders are often associated with the autoencoder model because of its architectural affinity, but with significant differences in the goal and mathematical formulation. Variational autoencoders allow statistical inference problems to be rewritten as statistical optimization problems. They are meant to map the input variable to a multivariate latent distribution. Although this type of model was initially designed for unsupervised learning, its effectiveness has been proven for semi-supervised learning and supervised learning.([Wikipedia](https://en.wikipedia.org/wiki/Variational_autoencoder))