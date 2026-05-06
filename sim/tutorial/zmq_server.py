import zmq

context = zmq.Context()
print("Starting ZMQ publisher...")
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

# while True:
#     # Modified: Added terminal input for target pose
#     try:
#         input_str = input("Enter target pose as 7 floats [x,y,z,qx,qy,qz,qw]: ")
#         parts = input_str.strip().split()
#         if len(parts) != 7:
#             print("Invalid input: need exactly 7 numbers")
#             continue
#         pose = [float(p) for p in parts]
#         msg = " ".join(str(p) for p in pose)
#         socket.send_string(msg)
#         print(f"Published pose: {msg}")
#     except ValueError:
#         print("Invalid input: not all numbers")
#     except KeyboardInterrupt:
#         break

while True:
    try:
        # 输入两只机械臂的目标位姿，每个 8 个数字
        input_str = input(
            "Enter target pose for 2 arms as 16 floats [x,y,z,qx,qy,qz,qw,gripper,... for arm1 and arm2]: "
        )
        parts = input_str.strip().split()
        
        if len(parts) != 16:
            print(f"Invalid input: need exactly 16 numbers, got {len(parts)}")
            continue
        
        # 转 float
        pose = [float(p) for p in parts]
        
        # 转成消息字符串
        msg = " ".join(str(p) for p in pose)
        
        # 发布消息
        socket.send_string(msg)
        print(f"Published pose: {msg}")
        
    except ValueError:
        print("Invalid input: not all numbers")
    except KeyboardInterrupt:
        print("Exiting publisher...")
        break