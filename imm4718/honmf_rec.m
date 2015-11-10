function Rec = honmf_rec(Core,FACT)

% Reconstructs the data from the Core and Factors found in the HONMF model
%
% Written by Morten Mørup
%
% Usage:
%   Rec = honmf_rec(Core,FACT)
%
% Input:
%   Core    The Core array of the TUCKER model
%   FACT    cell array of factors, FACT{i} pertains to i'th modality
%
% Output:
%   Rec     The reconstructed data
%
% Copyright (C) Morten Mørup and Technical University of Denmark, 
% September 2006
%                                          
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Rec=Core;
for i=1:length(FACT)
    Rec=tmult(Rec,FACT{i},i);
end