const SocketServer = require('websocket').server
const http = require('http')
const config = require('./config')
const CODES = require('./public_html/codes')
const express = require('express')
const app = express();
const cors = require('cors')
const log = require('./utils').log

const url_websocket =  'ws://'+config.ip+':'+config.port_socket+'/'
app.use(cors())

app.listen(config.port_express, function(){
    log("APP-EXPRESS", "La aplicación está escuchando en http://" + config.ip + ':' + config.port_express);
});


app.use(express.static(__dirname + "/public_html"));

const server = http.createServer((req, res) => {
		log("HTTP-SERVER", 'Received request for ' + req.url);
		res.writeHead(404);
		res.end();
})

server.listen(config.port_socket, ()=>{
    log("HTTP-SERVER", "Listening on port " + config.port_socket + "... Web socket listening in the same port")
})


const wsServer = new SocketServer({
	httpServer:server,
	autoAcceptConnections: false
	})

function originIsAllowed(origin) {
  // put logic here to detect whether the specified origin is allowed.
  return true;
}


const connections = []
const web_connections = []
const web_app_tunnel = {}  // las llaves son los rooms porque son únicos


const CONTEXT_WEB_SOCKET_SERVER = 'WEBSOCKET_SERVER'

var webConnection = null;


wsServer.on('request', (req) => {
	log(CONTEXT_WEB_SOCKET_SERVER, 'Connection in accepted from ' + req.remoteAddress);

	if (!originIsAllowed(req.origin)) {
      req.reject();
      log(CONTEXT_WEB_SOCKET_SERVER, 'Connection from origin ' + req.origin + ' rejected.');
      return;
    }



    var connection = req.accept('', req.origin);  // req.accept('echo-protocol', req.origin); req.accept();
    connections.push(connection)

    connection.on('message', (msg) => {
        
			var type = msg.type


			if(type == 'utf8'){

				var data = JSON.parse(msg.utf8Data)

				
                if(data.id == 0){
					if(webConnection != null){
						sendResponseToActualConnection(webConnection, msg.utf8Data)
					}else{
						sendResponseToActualConnection(connection, "LA WEB NO ESTA CONECTADA")
					}
				}

				else if(data.id == CODES.AUTH){
					var room = 10000// parseInt(Math.random() * 10000);
					web_app_tunnel[room] = [connection, null];
					sendResponseToActualConnection(connection, 
						response(1, "", -data.id, "int->"+room) )
				}
				else if(data.id == CODES.BEGIN_STREAM){

					var room = parseInt(data.metadata.split("->")[1])
					if(web_app_tunnel[room][1] != null){
						sendResponseToActualConnection(web_app_tunnel[room][1], 
							response(1, "", data.id,  data.data) )
					}
					else{
						sendResponseToActualConnection(connection, 
							response(1, "", -data.id, "string->MATLAB NO CONECTED") )
					}
				}
				else if(data.id == CODES.END_STREAM){

				}
				else if(data.id == CODES.PREDICTION){
					var room = parseInt(data.metadata.split("->")[1]);
					if(web_app_tunnel[room] == undefined){
						sendResponseToActualConnection(connection, 
							response(-1, "", -data.id, "string->Open a browser session first") )
						
					}
					else
					{
						sendResponseToActualConnection(web_app_tunnel[room][0], 
							response(1, "", data.id, data.data) )
					}
					
				}
				else if(data.id == CODES.ANSWER_PREDICTION){
					var room = parseInt(data.metadata.split("->")[1])
					if(web_app_tunnel[room][1] != null){
						sendResponseToActualConnection(web_app_tunnel[room][1], 
							response(1, "", -CODES.PREDICTION,  data.data) )
					}
					else{
						sendResponseToActualConnection(connection, 
							response(1, "", -data.id, "string->MATLAB NO CONECTED") )
					}
				}
				else if(data.id == CODES.CONNECT_ROOM){
					// message from MATLAB
					var room = parseInt(data.data.split("->")[1]);


					if(web_app_tunnel[room] == undefined){
						sendResponseToActualConnection(connection, 
							response(-1, "", -data.id, "string->Open a browser session first") )
						
					}
					else
					{
						web_app_tunnel[room][1] = connection;
						sendResponseToActualConnection(connection, 
							response(1, "", -data.id, "string->CONNECTED to room") )
					}
					
				}
				else if(data.id == CODES.PAIR_FUNCTION){

					var room = parseInt(data.metadata.split("->")[1])

					if(web_app_tunnel[room][1] != null){
						sendResponseToActualConnection(web_app_tunnel[room][1], 
							response(1, "", CODES.PAIR_FUNCTION, data.data) )
						
					}
					else{
						sendResponseToActualConnection(connection, 
							response(-1, "El cliente no está conectado al room", -CODES.PAIR_FUNCTION, "") )
					}

				}
				else if(data.id == CODES.CHART){
					var room = parseInt(data.metadata.split("->")[1]);

					if(web_app_tunnel[room] == undefined){
						sendResponseToActualConnection(connection, 
							response(-1, "", -data.id, "string->Open a browser session first") )
						
					}
					else
					{

						sendResponseToActualConnection(web_app_tunnel[room][0], 
							response(1, "Chart", data.id, data.data) )
					}
				}
				
				else{
					console.log("MESSAGE RECEIVED utf8: ", data)
				}


			}
			else{
				log(CONTEXT_WEB_SOCKET_SERVER, "MessageReceived, " + msg.utf8Data)
				sendResponseToActualConnection(connection, msg.utf8Data)
			}

    })

    connection.on('close', (resCode, des) => {
        log(CONTEXT_WEB_SOCKET_SERVER, 'connection closed ' + resCode + ' ' + des)
        connections.splice(connections.indexOf(connection), 1)

				if(web_connections.includes(connection)){
					// console.log(web_app_tunnel, web_rooms_associated)


					const web_index = web_connections.indexOf(connection)
					web_connections.splice(web_index, 1)

					log(CONTEXT_WEB_SOCKET_SERVER, "Removida la conexión WEB y rooms asociadas")

					// console.log(web_app_tunnel, web_rooms_associated)
				}
    })

})

function response(id, server_message, client_id, data){
	return JSON.stringify(
		{
			id: id,  // id of message or status. 0 is reserved for error
			server_message: server_message,
			metadata: client_id,
			data: data  //type is for clients to convert the string

		}
	)
}

function sendResponseToActualConnection(connection, myresponse){
	connections.forEach(element => {
		if (element == connection){
			element.sendUTF(JSON.stringify(myresponse))
			if(myresponse.length < 100){

				log(CONTEXT_WEB_SOCKET_SERVER, "Sending=> " + JSON.stringify(myresponse))
			}
		}
	})
}