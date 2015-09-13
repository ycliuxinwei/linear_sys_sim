function x = Sys(x,u,other,A,B)
	x = A*x+B*u+other;
end