function j = readjsonfile(f)
stuff = readtxtfile(f);
j = jsondecode(stuff{1});