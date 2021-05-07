function writejsonfile(s,jfile)
jse = jsonencode(s);
fh = fopen(jfile,'w');
fprintf(fh,'%s\n',jse);
fclose(fh);
fprintf(1,'Saved %s\n',jfile);