
var CODES = {AUTH: 1001};

load('../codes').then((LOADED)=>{
    CODES = LOADED;
})


const IP = "127.0.0.1";
const PORT = 3000;
var room_id = -1;






function getData() {
  return Math.random();
} 

var intervalo = null;



function estatico(){
    $("#stream").hide();
    $("#est").show();
}

function stream(){
    $("#stream").show();
    $("#est").hide();
}

function responseCustom(id, data){
    return JSON.stringify({
        id: id,
        metadata: "int->"+room_id,
        data: data
    })
}



/* La respuesta es el negativo del id:
1001 -> opening. metadata: string->ssssss, data -> any. Response expected: ROOM_ID
*/


const socket = new WebSocket('ws://'+IP+":"+PORT, 'echo');
socket.onopen = function () {
    socket.send(responseCustom(CODES.AUTH, "string->Opening connection in browser"));
};


function beginStream(){
  socket.send(responseCustom(CODES.BEGIN_STREAM, "string->begin"));

  Plotly.plot('chart',[{x: 0,
    y:[0],
    type:'line'
  }]);

}

function endStream()
{
  socket.send(responseCustom(CODES.END_STREAM, "string->detener"));
  // clearInterval(intervalo);
}

socket.onmessage = function (response) {


    message = JSON.parse(JSON.parse(response.data));


    if(message.id == CODES.ERROR){
        console.log("error: ", message.server_message)
    }


    else if(message.metadata == -CODES.AUTH){
        room_id = parseInt(message.data.split("->")[1])
        $("#room_info").text("ROOM: " + room_id)
    }
    else if(message.metadata == -CODES.PAIR_FUNCTION){
        $("#server_message").text(message.server_message)
    }
    else if(message.metadata == CODES.CHART){
      var matrix = JSON.parse(message.data.split("->")[1])

      
      var cnt = matrix.length;
      var maximo = 200;
      const data_y = matrix[cnt-1][0];
      
      if(cnt > maximo) {
        Plotly.relayout('chart',{
          xaxis: {
            range: [0, cnt+maximo]
          }
        });
      }
      Plotly.extendTraces('chart',{ y:[[data_y]]}, [0]);


    }
    else if(message.metadata == CODES.PREDICTION){
      var gesture = message.data.split("->")[1];

      var correct = confirm("¿Fue: " + gesture + "?")

      if(!correct){
        var correct_gesture = prompt("Correcto: ", "noGesture");
        socket.send(responseCustom(CODES.ANSWER_PREDICTION, "string->"+correct_gesture));
      }


    }else{

      console.log("RECEIVED", response.data);
    }

};

socket.onerror = function (error) {
    console.log('WebSocket error: ', error);
};


var chart = new CanvasJS.Chart("chartContainer1", {
  animationEnabled: true,
  theme: "light2",
  title:{
    text: "Simple Line Chart"
  },
  data: [{        
    type: "line",
        indexLabelFontSize: 16,
    dataPoints: [
      { y: 450 },
      { y: 414},
      { y: 520, indexLabel: "\u2191 highest",markerColor: "red", markerType: "triangle" },
      { y: 460 },
      { y: 450 },
      { y: 500 },
      { y: 480 },
      { y: 480 },
      { y: 410 , indexLabel: "\u2193 lowest",markerColor: "DarkSlateGrey", markerType: "cross" },
      { y: 500 },
      { y: 480 },
      { y: 510 }
    ]
  }]
});
chart.render();

