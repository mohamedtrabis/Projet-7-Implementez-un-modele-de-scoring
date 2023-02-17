<div class='titre_h'style='text-shadow:0 0 2px #000,0 0 30px #000,0px 0px 5px #000, 0 0 150px #000;color:#fff;font-weight:bold;background: #fff;/*border: solid #F6E5CB 1px;*/padding: 10px;border-radius:0 4px 4px 4px;-moz-border-radius: 0px 4px 4px 4px;-webkit-border-radius: 0px 4px 4px 4px;/*border: solid #F67F2B 15px;*/background-color:#fff;text-align:center;'>
  <h1>Implémentez un modèle de scoring</h1>
<br>
</div>
<p align="center" style='text-align:center;'><img src="https://bfsi.eletsonline.com/wp-content/uploads/2015/09/home-credit20120926124427_l.jpg" width="300" title="hover text"/>
</p>
<br>
<form>
  <fieldset>
    <legend class="alert alert-success" role="alert" style='margin:0 auto;'>
    <h4><b>Introduction :</b></h4>
    </legend>
    <br>
    <p class='tab2 active'><strong><span class="alert alert-info" role="alert">Contexte :</span></strong> </p>
    <br>
    <p>L’entreprise «HOME CREDIT» souhaite mettre en oeuvre un outil de «scoring crédit» pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).<br>
      Prêt à dépenser décide de développer un Dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.</p>
    <br>
    <p class='tab2 active'><strong><span class="alert alert-info" role="alert">Mission :</span></strong></p>
    <br>
    <div class='description2'>
      <p align='left'><img src='https://www.icone-png.com/png/29/29379.png' width='20px' height='20px' align='top' /> Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.</p>
      <p align='left'><img src='https://www.icone-png.com/png/29/29379.png' width='20px' height='20px' align='top' /> Construire un Dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.</p>
    </div>
  </fieldset>
</form>
</br>
<form>
  <fieldset>
    <legend class="alert alert-success" role="alert" style='margin:0 auto;'>
    <h4><b>Découverte des données:</b></h4>
    </legend>
    <br>
    <table width='100%' class='product-attribute-specs-table' id='product-attribute-specs-table'>
      <tr class='first odd'>
        <th class='label' style='background-color:#ECEBEB; text-align:left;font-weight:bold' width='30%'>application_train.csv</th>
        <td class='data last' width='50%'>Les principales données de formation avec des informations sur chaque demande de prêt chez Prêt à dépenser.</td>
      </tr>
      <tr class='even'>
        <th width='30%' class='label' style='background-color:#ECEBEB; text-align:left'>bureau.csv</th>
        <td width='30%' class='data last'>Données concernant les crédits antérieurs du client auprès d'autres institutions financières.</td>
      </tr>
      <tr class='last odd'>
        <th width='30%' class='label' style='background-color:#ECEBEB; text-align:left'>bureau_balance.csv</th>
        <td width='30%' class='data last'>Données mensuelles détaillées sur les crédits précédents dans le fichier bureau.csv</td>
      </tr>
      <tr class='last odd'>
        <th class='label' style='background-color:#ECEBEB; text-align:left'>credit_card_balance.csv</th>
        <td class='data last'>Données mensuelles sur les cartes de crédit précédentes que les clients ont eues avec Prêt à dépenser.</td>
      </tr>
      <tr class='last odd'>
        <th class='label' style='background-color:#ECEBEB; text-align:left'>installments_payments.csv</th>
        <td class='data last'>Historique de paiement pour les prêts précédents chez Prêt à dépenser.</td>
      </tr>
      <tr class='last odd'>
        <th width='30%' class='label' style='background-color:#ECEBEB; text-align:left'>previous_application.csv</th>
        <td width='30%' class='data last'>Demandes précédentes de prêts chez Prêt à dépenser des clients qui ont des prêts dans le fichier application_train.csv</td>
      </tr>
      <tr class='last odd'>
        <th width='30%' class='label' style='background-color:#ECEBEB; text-align:left'>POS_CASH_balance.csv</th>
        <td width='30%' class='data last'>Données mensuelles sur les clients précédents.</td>
      </tr>
    </table>
  </fieldset>
</form>
