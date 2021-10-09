function index_in_packet = getUserIndexInPacket(dataPacket, user_name)
%GETUSERINDEXINPACKET Summary of this function goes here
%   Detailed explanation goes here

for index_packet=1:numel(dataPacket)
   if dataPacket(index_packet).name == user_name
       index_in_packet = index_packet;
       break;      
   end
end

end

