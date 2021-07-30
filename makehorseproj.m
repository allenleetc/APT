function makehorseproj(varargin)
mov = 'horse.mov';
lObj = StartAPT;
cfg = Labeler.cfgGetLastProjectConfigNoView;
cfg.NumLabelPoints = 3;
cfg.NumViews = 1;
cfg.MultiAnimal = 0;
lObj.initFromConfig(cfg);
lObj.projNew('horse');
lObj.movieAdd(mov);
lObj.movieSet(1);
lObj.setFrame(5);
lObj.labelPosSet([40 50;50 60;80 100]);
lblname = fullfile(pwd,'horseproj.lbl');
lObj.projSaveRaw(lblname);
disp(lObj.projFSInfo.filename);
