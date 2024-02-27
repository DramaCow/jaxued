# Welcome to JaxUED!

JaxUED is a Unsupervised Environment Design library with similar goals to [CleanRL](https://docs.cleanrl.dev): high-quality, single-file and understandable implementations of common UED methods.

## What We Provide
JaxUED has several utilities that are useful for implementing UED methods, a `LevelSampler`, a general environment interface `UnderspecifiedEnv`, and a Maze implementation. 

We also have understandable single-file implementations of DR, PLR, ACCEL and PAIRED.

## Who JaxUED is for
JaxUED is primarily intended for researchers looking to get *in the weeds* of UED algorithm development. Our minimal dependency implementations of the current state-of-the art UED methods expose all implementation details; helping researchers understand how the algorithms work in practise, and facilitating easy, rapid prototyping of new ideas. 

## Why JaxUED?
- Single-file reference implementations of common-UED algorithms
- Understandable code with low levels of obscurity/abstraction.
- Wandb integration and logging of metrics and generated levels