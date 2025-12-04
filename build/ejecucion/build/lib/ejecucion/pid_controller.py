class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(-1.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        
        self.prev_error = 0
        self.integral = 0
    
    def compute(self, error, dt=0.1):
        # Proporcional
        P = self.Kp * error
        
        # Integral
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivativo
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative
        
        # Salida total
        output = P + I + D
        
        # Limitar salida
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        self.prev_error = error
        return output
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0

