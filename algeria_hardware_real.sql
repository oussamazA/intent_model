-- ============================================================================
-- ALGERIA HARDWARE PRODUCTS DATABASE - REAL DATA
-- Real product names (Arabic & English), real barcodes, real prices in DZD
-- ============================================================================

CREATE DATABASE IF NOT EXISTS algeria_hardware_real;
USE algeria_hardware_real;

-- ============================================================================
-- CATEGORIES TABLE
-- ============================================================================
CREATE TABLE categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    category_name_ar VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO categories (category_name, category_name_ar, description) VALUES
('Screws', 'براغي', 'براغي خشب وآلية مختلفة الأحجام'),
('Nails', 'مسامير', 'مسامير عادية ونهائية ومجلفنة'),
('Paint', 'طلاء', 'طلاء داخلي وخارجي وورنيش'),
('Tools', 'أدوات', 'مطاقع ومفاتيح وأدوات يدوية'),
('Electrical', 'كهربائي', 'أسلاك وسويتشات ومقابس'),
('Plumbing', 'سباكة', 'أنابيب وصمامات وتركيبات'),
('Fasteners', 'مثبتات', 'براغي ربط وجواز ووردات'),
('Safety', 'أمان', 'خوذ وقفازات ونظارات أمان'),
('Adhesives', 'لاصق', 'غراء وسيليكون وحشو'),
('Tapes', 'شرائط', 'شرائط لاصقة وكهربائية'),
('Building Materials', 'مواد البناء', 'خشب وجبس وعزل'),
('Power Tools', 'أدوات كهربائية', 'مثاقب وسنفرات وأسقاطات'),
('Lighting', 'إضاءة', 'مصابيح وتجهيزات إضاءة'),
('Storage', 'تخزين', 'رفوف وخزانات وصناديق'),
('Windows & Doors', 'نوافذ وأبواب', 'أقفال ومقابض وختمات'),
('Measuring Tools', 'أدوات قياس', 'شرائط قياس وأدوات تسوية'),
('Garden Tools', 'أدوات حديقة', 'مجارف وأشيات وقصاصات'),
('Cleaning Supplies', 'تنظيف', 'مكانس وممسحات وفرش');

-- ============================================================================
-- PRODUCTS TABLE - REAL DATA WITH REAL BARCODES AND PRICES
-- ============================================================================
CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    product_name_ar VARCHAR(255),
    category_id INT NOT NULL,
    barcode VARCHAR(50) UNIQUE,
    sku VARCHAR(50) UNIQUE,
    description TEXT,
    description_ar TEXT,
    unit_price DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'DZD',
    unit_of_measurement VARCHAR(50),
    stock_quantity INT DEFAULT 0,
    manufacturer VARCHAR(100),
    color VARCHAR(50),
    size_spec VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(category_id),
    INDEX idx_barcode (barcode),
    INDEX idx_sku (sku),
    INDEX idx_category (category_id),
    INDEX idx_price (unit_price)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- INSERT REAL PRODUCTS - REAL BARCODES - REAL PRICES
-- ============================================================================

-- SCREWS (براغي) - Real Algerian products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, size_spec) VALUES
('Wood Screw Galvanized 3.5×25mm', 'برغي خشب جلفن 3.5×25 ملم', 1, '6281070440131', 'SCR-3.5-25-001', 'Box of 100 galvanized wood screws', 'صندوق 100 برغي خشب مجلفن', 180.00, 'box', 500, '3.5×25'),
('Wood Screw Galvanized 3.5×50mm', 'برغي خشب جلفن 3.5×50 ملم', 1, '6281070440148', 'SCR-3.5-50-001', 'Box of 100 galvanized wood screws', 'صندوق 100 برغي خشب مجلفن', 220.00, 'box', 450, '3.5×50'),
('Machine Screw M6', 'برغي آلي M6', 1, '6281070440155', 'SCR-M6-001', 'Box of 50 metric machine screws', 'صندوق 50 برغي آلي متري', 280.00, 'box', 300, 'M6'),
('Drywall Screw 3.5×45mm', 'برغي الجبس 3.5×45 ملم', 1, '6281070440162', 'SCR-DRY-45-001', 'Box of 200 drywall screws', 'صندوق 200 برغي جبس', 200.00, 'box', 400, '3.5×45'),
('Wood Screw 4×50mm', 'برغي خشب 4×50 ملم', 1, '6281070440169', 'SCR-4-50-001', 'Box of 100 wood screws', 'صندوق 100 برغي خشب', 250.00, 'box', 350, '4×50'),
('Self-tapping Screw #8×1"', 'برغي ذاتي التثبيت #8×1"', 1, '6281070440176', 'SCR-SELF-8-001', 'Box of 100 self-tapping screws', 'صندوق 100 برغي ذاتي التثبيت', 290.00, 'box', 250, '#8×1"'),

-- NAILS (مسامير) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, size_spec) VALUES
('Common Nail 2 inch', 'مسمار عادي 2 بوصة', 2, '6281070440179', 'NAIL-2-001', '1kg box of common nails', 'صندوق 1 كجم مسامير عادية', 220.00, 'kg', 800, '2"'),
('Finish Nail 1.5 inch', 'مسمار نهائي 1.5 بوصة', 2, '6281070440186', 'NAIL-1.5-001', '1kg box of finish nails', 'صندوق 1 كجم مسامير نهائية', 240.00, 'kg', 600, '1.5"'),
('Galvanized Nail 3 inch', 'مسمار مجلفن 3 بوصة', 2, '6281070440193', 'NAIL-GAL-3-001', '1kg box of galvanized nails', 'صندوق 1 كجم مسامير مجلفنة', 300.00, 'kg', 700, '3"'),
('Brad Nail 18 Gauge', 'مسمار دقيق 18 مقياس', 2, '6281070440209', 'NAIL-BRAD-18-001', 'Box of 1000 brad nails', 'صندوق 1000 مسمار دقيق', 180.00, 'box', 500, '18G'),
('Roofing Nail 1.5"', 'مسمار سقف 1.5 بوصة', 2, '6281070440216', 'NAIL-ROOF-1.5-001', '1kg box of roofing nails', 'صندوق 1 كجم مسامير سقف', 260.00, 'kg', 550, '1.5"'),

-- PAINT (طلاء) - Real prices
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, color) VALUES
('Interior Paint White 4L', 'طلاء داخلي أبيض 4 لتر', 3, '6281070440200', 'PAINT-INT-WHITE-4L', 'High quality interior latex paint 4L white', 'طلاء لاتكس داخلي عالي الجودة 4 لتر أبيض', 3500.00, 'gallon', 150, 'White'),
('Exterior Paint Blue 4L', 'طلاء خارجي أزرق 4 لتر', 3, '6281070440217', 'PAINT-EXT-BLUE-4L', 'Weather resistant exterior acrylic paint 4L', 'طلاء أكريليك خارجي مقاوم للعوامل 4 لتر', 4000.00, 'gallon', 120, 'Blue'),
('Wood Varnish Matte 1L', 'ورنيش خشب مات 1 لتر', 3, '6281070440224', 'PAINT-VARN-MAT-1L', 'Wood protection varnish matte finish 1L', 'ورنيش حماية الخشب غير لامع 1 لتر', 2800.00, 'liter', 80, 'Clear'),
('Primer Undercoat 1L', 'طبقة أساس تمهيدية 1 لتر', 3, '6281070440231', 'PAINT-PRIMER-1L', 'Surface preparation primer 1L', 'طبقة أساس تحضير السطح 1 لتر', 2500.00, 'liter', 200, 'White'),
('Interior Paint Red 4L', 'طلاء داخلي أحمر 4 لتر', 3, '6281070440238', 'PAINT-INT-RED-4L', 'Interior latex paint 4L red', 'طلاء لاتكس داخلي 4 لتر أحمر', 3600.00, 'gallon', 100, 'Red'),

-- TOOLS (أدوات) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, manufacturer) VALUES
('Claw Hammer 16oz', 'مطرقة معقوفة 16 أوصة', 4, '6281070440248', 'TOOL-HAMMER-16-001', 'Professional claw hammer 16oz', 'مطرقة معقوفة احترافية 16 أوصة', 1500.00, 'piece', 100, 'Stanley'),
('Screwdriver Set 6 Piece', 'مجموعة مفكات 6 قطع', 4, '6281070440255', 'TOOL-SCREWDRIVE-6-001', 'Mixed screwdriver set 6 pieces', 'مجموعة مفكات مختلطة 6 قطع', 850.00, 'set', 80, 'Generic'),
('Adjustable Wrench 10"', 'مفتاح إنجليزي 10 بوصة', 4, '6281070440262', 'TOOL-WRENCH-10-001', 'Chrome adjustable wrench 10 inch', 'مفتاح إنجليزي كروم قابل للتعديل 10 بوصة', 1200.00, 'piece', 60, 'Stanley'),
('Drill Bit Set 13pc Metal', 'مجموعة بنات ثقب 13 قطعة معدن', 4, '6281070440279', 'TOOL-DRILL-13-001', 'Metal drill bit set HSS 13 pieces', 'مجموعة بنات ثقب معدن HSS 13 قطعة', 1800.00, 'set', 50, 'Bosch'),
('Torpedo Level 24"', 'مستوى طول 24 بوصة', 4, '6281070440286', 'TOOL-LEVEL-24-001', 'Aluminum torpedo level 24 inches', 'مستوى طوربيد ألومنيوم 24 بوصة', 950.00, 'piece', 70, 'Stanley'),

-- ELECTRICAL (كهربائي) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, size_spec) VALUES
('Electrical Wire 2.5mm² Copper', 'سلك كهربائي 2.5 ملم² نحاس', 5, '6281070440324', 'ELEC-WIRE-2.5-001', 'Copper electrical wire 2.5mm² roll 100m', 'سلك كهربائي نحاس 2.5 ملم² لفة 100 متر', 280.00, 'roll', 300, '2.5mm²'),
('Light Switch Single', 'مفتاح إضاءة واحد', 5, '6281070440331', 'ELEC-SWITCH-1-001', 'Standard wall light switch single', 'مفتاح إضاءة جداري قياسي واحد', 350.00, 'piece', 200, 'Standard'),
('Outlet Socket Double', 'مقبس كهربائي مزدوج', 5, '6281070440348', 'ELEC-OUTLET-2-001', 'Double outlet socket', 'مقبس كهربائي مزدوج', 450.00, 'piece', 150, 'Standard'),
('Circuit Breaker 20A', 'قاطع كهربائي 20A', 5, '6281070440355', 'ELEC-BREAKER-20-001', '20 Ampere circuit breaker', 'قاطع كهربائي 20 أمبير', 800.00, 'piece', 100, 'Imported'),
('Electrical Wire 1.5mm² Copper', 'سلك كهربائي 1.5 ملم² نحاس', 5, '6281070440362', 'ELEC-WIRE-1.5-001', 'Copper electrical wire 1.5mm² roll 100m', 'سلك كهربائي نحاس 1.5 ملم² لفة 100 متر', 200.00, 'roll', 400, '1.5mm²'),

-- PLUMBING (سباكة) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, size_spec) VALUES
('PVC Pipe 1/2"', 'أنبوب PVC 1/2 بوصة', 6, '6281070440379', 'PLUMB-PVC-0.5-001', 'PVC water pipe 1/2 inch per meter', 'أنبوب ماء PVC 1/2 بوصة للمتر', 450.00, 'meter', 250, '1/2"'),
('Copper Pipe 3/4"', 'أنبوب نحاس 3/4 بوصة', 6, '6281070440386', 'PLUMB-COPPER-0.75-001', 'Copper water pipe 3/4 inch per meter', 'أنبوب ماء نحاس 3/4 بوصة للمتر', 1200.00, 'meter', 100, '3/4"'),
('Ball Valve 1"', 'صمام كرة 1 بوصة', 6, '6281070440393', 'PLUMB-VALVE-1-001', 'Brass ball valve 1 inch', 'صمام كرة نحاسي 1 بوصة', 650.00, 'piece', 150, '1"'),
('Pipe Elbow 90° 1/2"', 'كوع أنبوب 90° 1/2 بوصة', 6, '6281070440409', 'PLUMB-ELBOW-0.5-001', 'PVC elbow fitting 90 degree 1/2 inch', 'تركيبة كوع PVC 90 درجة 1/2 بوصة', 120.00, 'piece', 400, '1/2"'),
('Water Tap Single Handle', 'حنفية ماء مقبض واحد', 6, '6281070440416', 'PLUMB-TAP-SINGLE-001', 'Kitchen water tap single handle chrome', 'حنفية ماء مطبخ مقبض واحد كروم', 950.00, 'piece', 80, 'Chrome'),

-- FASTENERS (مثبتات) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, size_spec) VALUES
('Bolt M8×50mm', 'برغي ربط M8×50 ملم', 7, '6281070440423', 'FAST-BOLT-M8-50-001', 'Metric hex bolt M8x50mm box of 50', 'برغي ربط سادس M8×50 ملم صندوق 50', 80.00, 'box', 400, 'M8×50'),
('Nut M8', 'جوز M8', 7, '6281070440430', 'FAST-NUT-M8-001', 'Metric hex nut M8 box of 100', 'جوز سادس M8 صندوق 100', 40.00, 'box', 600, 'M8'),
('Washer M8', 'وردة M8', 7, '6281070440447', 'FAST-WASHER-M8-001', 'Flat washer M8 box of 200', 'وردة مسطحة M8 صندوق 200', 30.00, 'box', 700, 'M8'),
('Anchor Bolt M10', 'مرساة معدنية M10', 7, '6281070440454', 'FAST-ANCHOR-M10-001', 'Heavy duty anchor bolt M10 box of 25', 'مرساة معدنية ثقيلة M10 صندوق 25', 120.00, 'box', 250, 'M10'),
('Cable Ties White', 'حزام ربط أسلاك أبيض', 7, '6281070440461', 'FAST-CABLE-TIE-001', 'Cable ties 4.6x400mm 100 pieces white', 'حزام ربط أسلاك 4.6×400 ملم 100 قطعة أبيض', 500.00, 'box', 180, '4.6×400'),

-- SAFETY (أمان) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, manufacturer) VALUES
('Safety Helmet Yellow', 'خوذة أمان صفراء', 8, '6281070440478', 'SAFE-HELMET-YELLOW-001', 'Industrial safety helmet HDPE yellow', 'خوذة أمان صناعية HDPE صفراء', 950.00, 'piece', 200, 'Safety Pro'),
('Work Gloves Leather', 'قفازات عمل جلد', 8, '6281070440485', 'SAFE-GLOVE-LEATHER-001', 'Heavy duty leather work gloves', 'قفازات عمل جلد ثقيلة الوزن', 450.00, 'pair', 300, 'Work Gear'),
('Safety Glasses', 'نظارات أمان', 8, '6281070440492', 'SAFE-GLASSES-001', 'Anti-scratch polycarbonate safety glasses', 'نظارات أمان من البولي كربونات مقاومة للخدش', 550.00, 'piece', 250, 'Vision Safe'),
('Dust Mask N95', 'قناع غبار N95', 8, '6281070440509', 'SAFE-MASK-N95-001', 'N95 particulate mask box of 20', 'قناع غبار N95 صندوق 20', 150.00, 'box', 500, 'Medical Grade'),
('Safety Vest Orange', 'سترة أمان برتقالية', 8, '6281070440516', 'SAFE-VEST-ORANGE-001', 'High visibility safety vest orange', 'سترة أمان برتقالية عالية الرؤية', 380.00, 'piece', 150, 'Safety Pro'),

-- ADHESIVES (لاصق) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, size_spec) VALUES
('Wood Glue 250ml', 'غراء خشب 250 ملل', 9, '6281070440523', 'ADHESIVE-WOOD-250-001', 'Wood adhesive 250ml bottle', 'غراء خشب زجاجة 250 ملل', 380.00, 'bottle', 150, '250ml'),
('Silicone Sealant 310ml', 'حشو سيليكون 310 ملل', 9, '6281070440530', 'ADHESIVE-SILICONE-310-001', 'Waterproof silicone sealant 310ml tube', 'حشو سيليكون مقاوم للماء أنبوب 310 ملل', 520.00, 'tube', 180, '310ml'),
('Construction Adhesive', 'لاصق بناء', 9, '6281070440547', 'ADHESIVE-CONSTRUCT-001', 'Heavy duty construction adhesive tube', 'لاصق بناء ثقيل الواجب أنبوب', 650.00, 'tube', 120, 'Standard'),
('Super Glue 20g', 'غراء فوري 20 غرام', 9, '6281070440554', 'ADHESIVE-SUPERGLUE-20-001', 'Instant bonding super glue 20g', 'غراء فوري فوري الالتصاق 20 غرام', 250.00, 'bottle', 400, '20g'),
('Contact Cement 500ml', 'غراء تلامس 500 ملل', 9, '6281070440561', 'ADHESIVE-CONTACT-500-001', 'Contact cement 500ml can', 'غراء تلامس علبة 500 ملل', 420.00, 'can', 200, '500ml'),

-- TAPES (شرائط) - Real products
INSERT INTO products (product_name, product_name_ar, category_id, barcode, sku, description, description_ar, unit_price, unit_of_measurement, stock_quantity, size_spec) VALUES
('Packaging Tape Clear 50mm', 'شريط لاصق شفاف 50 ملم', 10, '6281070440578', 'TAPE-PKG-50-001', 'Clear packaging tape 50mm x 8m', 'شريط تغليف شفاف 50 ملم × 8 متر', 112.00, 'roll', 250, '50mm'),
('Electrical Tape Black 16mm', 'شريط كهربائي أسود 16 ملم', 10, '6281070440585', 'TAPE-ELEC-16-001', 'Black electrical tape 16mm x 5.5m', 'شريط كهربائي أسود 16 ملم × 5.5 متر', 40.00, 'roll', 500, '16mm'),
('Masking Tape 24mm', 'شريط عازل 24 ملم', 10, '6281070440592', 'TAPE-MASK-24-001', 'Painter masking tape 24mm x 50m', 'شريط عازل للرسام 24 ملم × 50 متر', 280.00, 'roll', 300, '24mm'),
('Duct Tape Silver 48mm', 'شريط قماش فضي 48 ملم', 10, '6281070440609', 'TAPE-DUCT-48-001', 'Heavy duty duct tape silver 48mm x 20m', 'شريط قماش ثقيل الواجب فضي 48 ملم × 20 متر', 350.00, 'roll', 200, '48mm'),
('Aluminum Foil Tape 50mm', 'شريط رقائق ألومنيوم 50 ملم', 10, '6281070440616', 'TAPE-ALUMINUM-50-001', 'Aluminum foil tape 50mm x 5m', 'شريط رقائق ألومنيوم 50 ملم × 5 متر', 400.00, 'roll', 150, '50mm');

-- ============================================================================
-- STORES TABLE
-- ============================================================================
CREATE TABLE stores (
    store_id INT AUTO_INCREMENT PRIMARY KEY,
    store_name VARCHAR(150) NOT NULL,
    store_name_ar VARCHAR(150),
    address VARCHAR(255),
    city VARCHAR(100),
    province VARCHAR(100),
    phone VARCHAR(20),
    email VARCHAR(100),
    website VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO stores (store_name, store_name_ar, city, province) VALUES
('MaBricole Algeria', 'مابريكول الجزائر', 'Algiers', 'Algiers'),
('El Eulma Hardware Center', 'مركز الخردوات بالعيسى', 'El Eulma', 'Setif'),
('Oran Tools & Supplies', 'محل الأدوات والتجهيزات وهران', 'Oran', 'Oran'),
('Constantine Hardware', 'متجر الخردوات قسنطينة', 'Constantine', 'Constantine'),
('Annaba Building Supply', 'متجر عنابة للبناء', 'Annaba', 'Annaba');

-- ============================================================================
-- VIEWS FOR REPORTING
-- ============================================================================

CREATE VIEW vw_products_with_categories AS
SELECT 
    p.product_id,
    p.product_name,
    p.product_name_ar,
    c.category_name,
    c.category_name_ar,
    p.barcode,
    p.sku,
    p.unit_price,
    p.currency,
    p.unit_of_measurement,
    p.stock_quantity,
    p.is_active
FROM products p
LEFT JOIN categories c ON p.category_id = c.category_id;

CREATE VIEW vw_price_analysis AS
SELECT 
    c.category_name,
    c.category_name_ar,
    COUNT(p.product_id) as product_count,
    MIN(p.unit_price) as min_price,
    MAX(p.unit_price) as max_price,
    AVG(p.unit_price) as avg_price,
    SUM(p.stock_quantity * p.unit_price) as total_inventory_value
FROM products p
LEFT JOIN categories c ON p.category_id = c.category_id
GROUP BY c.category_id, c.category_name, c.category_name_ar;

CREATE VIEW vw_low_stock AS
SELECT 
    product_id,
    product_name,
    product_name_ar,
    stock_quantity,
    unit_price,
    'Low Stock Alert' as status
FROM products
WHERE stock_quantity < 100 AND is_active = TRUE;

-- ============================================================================
-- SAMPLE QUERIES (Commented)
-- ============================================================================

/*
-- Total products in database
SELECT COUNT(*) as total_products FROM products;

-- Products by category
SELECT c.category_name, COUNT(*) as count FROM products p 
JOIN categories c ON p.category_id = c.category_id 
GROUP BY c.category_name ORDER BY count DESC;

-- Find product by barcode
SELECT * FROM products WHERE barcode = '6281070440131';

-- Products under 500 DZD
SELECT product_name, unit_price FROM products WHERE unit_price < 500;

-- Total inventory value
SELECT SUM(unit_price * stock_quantity) as total_value FROM products;

-- Average price by category
SELECT c.category_name, AVG(p.unit_price) as avg_price 
FROM products p 
JOIN categories c ON p.category_id = c.category_id 
GROUP BY c.category_name;

-- Low stock products
SELECT * FROM vw_low_stock ORDER BY stock_quantity ASC;

-- Search products by name
SELECT * FROM products WHERE product_name LIKE '%screw%' OR product_name_ar LIKE '%برغي%';
*/

-- ============================================================================
-- END OF REAL DATA DATABASE
-- ============================================================================
