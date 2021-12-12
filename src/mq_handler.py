
import pika
import uuid
class RpcClient(object):

    def __init__(self, host, routing_key, logger):
        self.host = host
        self.routing_key = routing_key
        
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.host))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)
        
        self.logger.info(f"[STOP] RPC request client initilized. Host={host}, Routing={routing_key}")

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, commannd_msg=None):
        commannd_msg = "" if commannd_msg is None else commannd_msg
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.logger(f"[REQUEST] Call RPC. ID={self.corr_id}, Command={commannd_msg}")
        self.channel.basic_publish(
            exchange='',
            routing_key=self.routing_key,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=str(commannd_msg))
        while self.response is None:
            self.connection.process_data_events()
        return self.response
    
    
    def end_call(self):
        self.call("END")


        
    
