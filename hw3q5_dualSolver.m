clc
clear all
% Script for solving Lagrange Multipliers
H = [25, 1, -1, -25; 1, 25, -1, -25; -1, -1, 1, 1; -25, -25, 1, 81];
f = [-1,-1,-1,-1];
Aeq = [-1,-1,1,1];
beq = 0;
lb = zeros(4,1);
alpha = quadprog(H,f,[],[],Aeq,beq,lb); 
disp(alpha)