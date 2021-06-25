# Stock-Index-Forecasting-for-Trading

# การสร้าง model 
- tdnn_model.ipynb สําหรับสร้าง model Time delay neural network
- tdnn_pso_model.ipynb สําหรับสร้าง model Time delay neural network ที่มี particle swarm optimization ช่วยเพิ่มประสิทธิภาพ
- rf.model.ipynb สําหรับสร้าง model Random forest 
- svm_model.ipynb สําหรับสร้าง model Support vector machines

# folder utils มีฟังก์ชั่นสําหรับปรับแต่งข้อมูลสําหรับการทําทดสอบต้นแบบพยากรณ์ และauto trading
- predictor.py เรียกใช้งาน model
- preprocessstock.py ดาวน์โหลดและจัดรูปข้อมูล
- progressbar.py สร้าง progressbar ตอนสร้างหรือเรียกใช้ model
- smoother.py ทํา moving average ข้อมูล
- batch_manager.py จัดการ batch

folder models ฟังก์ชั่นสําหรับการทํา particle swarm optimization

folder weights สําหรับเก็บบันทึก model

# การคัดเลือก Feature สําหรับสร้าง model
- feature_selection.ipynb ทําทดสอบการคัดเลือก feature จาก indicator จํานวน 84 ตัวโดยใช้ filter method และ wrapper method

# การทดสอบความถูกต้องของการพยากรณ์ของแต่ละ model
- test_predicted.ipynb
ใช้ Root Mean Square Error และ Explained Variance Score สําหรับการประเมิน model แต่ละตัว

# การทํา auto trade 
- simulator.ipynb
- simulator.py
ดู %change ของราคาปิดวันนี้กับราคาปิดที่พยากรณ์ได้ในอีก 6 วันข้างหน้า
ประเมินผลโดยใช้ ผลกําไรและ Sharpe Ratio
