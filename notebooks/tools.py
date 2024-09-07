import pandas as pd
import numpy as np

def visualize_data(dataFrame):
    data = pd.DataFrame({
        'Fetaure': dataFrame.columns.values,
        'Tipo': dataFrame.dtypes.values,
        'Nulos (%)': dataFrame.isna().mean().values * 100,
        'Negativos (%)': [len(dataFrame[col][dataFrame[col] < 0]) / len(dataFrame) * 100 if col in dataFrame.select_dtypes(include=[np.number]).columns else 0 for col in dataFrame.columns],
        'Zeros (%)': [len(dataFrame[col][dataFrame[col] == 0]) / len(dataFrame) * 100 if col in dataFrame.select_dtypes(include=[np.number]).columns else 0 for col in dataFrame.columns], 
        'Duplicados': dataFrame.duplicated().sum(),
        'Unicos': dataFrame.nunique().values,
        'Valores unicos': [dataFrame[col].unique() for col in dataFrame.columns]
    })
    
    return data.round(2)

translations = {
    'order_id': 'ID do Pedido',
    'customer_id': 'ID do Cliente',
    'order_status': 'Status do Pedido',
    'order_purchase_timestamp': 'Data da Compra',
    'order_approved_at': 'Data de Aprovação do Pedido',
    'order_delivered_carrier_date': 'Data de Entrega ao Transportador',
    'order_delivered_customer_date': 'Data de Entrega ao Cliente',
    'order_estimated_delivery_date': 'Data Estimada de Entrega',
    'order_item_id': 'ID do Item do Pedido',
    'product_id': 'ID do Produto',
    'seller_id': 'ID do Vendedor',
    'shipping_limit_date': 'Data Limite de Envio',
    'price': 'Preço',
    'freight_value': 'Valor do Frete',
    'customer_unique_id': 'ID Único do Cliente',
    'customer_zip_code_prefix': 'Prefixo do CEP do Cliente',
    'customer_city': 'Cidade do Cliente',
    'customer_state': 'Estado do Cliente',
    'review_id': 'ID da Avaliação',
    'review_score': 'Nota da Avaliação',
    'review_comment_title': 'Título do Comentário da Avaliação',
    'review_comment_message': 'Mensagem do Comentário da Avaliação',
    'review_creation_date': 'Data de Criação da Avaliação',
    'review_answer_timestamp': 'Data da Resposta da Avaliação',
    'payment_sequential': 'Sequencial de Pagamento',
    'payment_type': 'Tipo de Pagamento',
    'payment_installments': 'Parcelas de Pagamento',
    'payment_value': 'Valor do Pagamento',
    'seller_zip_code_prefix': 'Prefixo do CEP do Vendedor',
    'seller_city': 'Cidade do Vendedor',
    'seller_state': 'Estado do Vendedor',
    'product_category_name': 'Categoria do Produto',
    'product_name_lenght': 'Comprimento do Nome do Produto',
    'product_description_lenght': 'Comprimento da Descrição do Produto',
    'product_photos_qty': 'Quantidade de Fotos do Produto',
    'product_weight_g': 'Peso do Produto (g)',
    'product_length_cm': 'Comprimento do Produto (cm)',
    'product_height_cm': 'Altura do Produto (cm)',
    'product_width_cm': 'Largura do Produto (cm)'
}