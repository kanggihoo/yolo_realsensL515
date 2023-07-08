from bboxes.srv import ImageFirstPickBox      # CHANGE
import sys
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge


class ServiceClient(Node):

    def __init__(self):
        super().__init__('srv_client')
        self.cli = self.create_client(ImageFirstPickBox, 'first_picks')       # CHANGE
        while not self.cli.wait_for_service(timeout_sec=1.0): # 1초 간격으로 service server동작 중인지 확인
            self.get_logger().info('service not available, waiting again...')
        self.req = ImageFirstPickBox.Request() # Service Request 정의  
        self.br = CvBridge()

    def send_request(self):
        future = self.cli.call_async(self.req)
        print("receive_request!!")
        return future


def ros_main(args=None):
    rclpy.init(args=args)

    clinet_node = ServiceClient()
    future = clinet_node.send_request() 

    while rclpy.ok():
        rclpy.spin_once(clinet_node)
        if future.done(): # 잘 전달 받은 경우 
            try:
                response = future.result() # service 결과값 저장(response )
                image1 =  clinet_node.br.imgmsg_to_cv2(response.color_image)


                cv2.imshow("color_image" , image1)
                if response.success is True:
                    image2 =  clinet_node.br.imgmsg_to_cv2(response.depth_image)
                    cv2.imshow("depth_image" , image2)
                cv2.waitKey(2000)
            # except Exception as e:
            #     clinet_node.get_logger().info(
            #         'Service call failed %r' % (e,))
            finally:
                clinet_node.get_logger().info(
                    f'Result of response x: {response.x}  y: {response.y}  z:{response.z} , label:{response.class_id}')
                    
                cv2.destroyAllWindows()
            break

    clinet_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    ros_main()
