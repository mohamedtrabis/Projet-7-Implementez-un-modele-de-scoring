<p align="center"><h1>Projet 7-Implementez un modele de scoring</h1></p>

<p align="center">
  <img src="https://github.com/mohamedtrabis/Projet-7-Implementez-un-modele-de-scoring/blob/main/Image/home credit.jpg" width="300" title="hover text">
</p>

## Contexte

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

## Mission

- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

## Les données
<p>Voici <a href="https://www.kaggle.com/c/home-credit-default-risk/data">les données</a> dont vous aurez besoin pour réaliser le dashboard. </p>
</p>Pour plus de simplicité, vous pouvez les télécharger à <a href="https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip">cette adresse</a>.</p>

<html>
<head>
<meta meta http-equiv='Content-Type' content='text/html; charset=UTF-8' />

<style>
body {
	font-family: 'Roboto', Arial, Helvetica, sans-serif;
}
#menu-principale, #motore-ricerca, .breadcrumb, /*#header,*/ .social, .condividi, .jcarousel-skin-tango, .button, .box-footer, #footer, .tab, .qty_buttons, .buy-button, .button-match, .no-rating, .etabs, .more-views, .discount_percentage, .menu-lingua, .button-newsletter, .account-logout, .cart, .qty, .qty_more, .qty_less, #footer-fine, .variante_colore, .consigli, .searchautocomplete, .nav, .btn, .btn-cart, .my-orders-table, .header-top-menu, .numero-verde, .nav, .register-login, .sale-banner-container,  /*.panel-container*/ #customer-reviews, .form-add, #product-reviews, .stampa, .ratings, .delivery, .ship {/*display: none !important;*/
}
.tab-container {
	background: #fff;/*border: solid #F6E5CB 1px;*/
	padding: 10px;
	border-radius:0 4px 4px 4px;
	-moz-border-radius: 0px 4px 4px 4px;
	-webkit-border-radius: 0px 4px 4px 4px;/*border: solid #F67F2B 15px;*/
*/
}
.bandeau-description {
	border: solid #8B0000 1px;
}
.tab-container .panel-container {
	background: #fff;/*border: solid #F6E5CB 15px;border: solid #F67F2B 15px;*/
	padding: 10px;
	border-radius:0 4px 4px 4px;
	-moz-border-radius: 0px 4px 4px 4px;
	-webkit-border-radius: 0px 4px 4px 4px;
}
#tab-description {
	color: #555555;
	font-size: 13px;
}
table#product-review-table.data-table {
	margin: 0;
	overflow: auto;
	width: 680px;
}
.panel-container form#review-form h3 {
	color: #666;
	display: block;
	font-family: 'arial';
	font-size: 15px;
	padding: 0 0 8px;
	font-weight:normal;
}
.panel-container #review-form fieldset h3 span {
	color: #666;
	font-family: lola;
	font-size: 15px;
	font-weight:normal;
}
#tab-description {
	color: #555555;
	font-size: 13px;
}
.product-description {
	color: #666;
	font-size: 13px;
	margin: 1em 0;
}
#my-reviews-table tr {
	width:100%!important;
	padding:0!important;
	transition:all ease-out 250ms;
}
#my-reviews-table th {
	width:132px!important;
	background:#8B0000;
}/*#my-reviews-table .even { background:#f8f8f8}*/
#my-events-table tr {
	width:658px!important;
	padding:0!important;
	border:none!important;
	height:45px;
}
#my-events-table .first {/*background:#8B0000;*/
	background-image:url(https://www.vosbesoinsinformatiques.com/wp-content/themes/blogolifepro/images/red/line.png);
	margin:0
}
#my-events-table th {
	width:117px!important
}
#my-events-table .even {
	margin: 0;
	padding: 8px 4px !important;
}
#my-events-table .pulsanti {/*background: none repeat scroll 0 0 #F7F7F7 !important;*/
	border-bottom: 1px solid #CCCCCC !important;/*border-radius: 4px !important;*/
	height: 40px !important;
	margin: 0 0 3px;
	padding: 0 !important;
}
#my-events-table .pulsanti a {
	color: #666 !important;/*font-family: lola;*/
	font-size: 14px;
	padding: 0 18px;
	font-weight:bold
}
#my-events-table thead tr {
	border: medium none !important;
	height: 38px;
	padding: 0 !important;
	width: 658px !important;
}
#my-events-table .pulsanti a:hover {
	text-decoration:underline;
}
#my-events-table a {/*color:#8B0000*/
	background-image:url(https://www.vosbesoinsinformatiques.com/wp-content/themes/blogolifepro/images/red/line.png);
}
.data-table tr {    /*float: left;*/
	border-radius:5px; /*line-height:15px;*/
}
.data-table .even {
	border-bottom: 1px solid #ccc;
	border-radius: 0 !important;
}
.data-table th {
	color: #FFFFFF;   /* display: block;    float: left;  */
	font-size: 12px;
	margin: 10px 0 0;
	width:64px;
}
.data-table tr {   /* float: left; */
	width: 387px;
	height:35px;
}
.first.odd > td {
	text-align: left;
	width: 110px;
	color: #666666;
	font-size: 12px;
}
.even > td {
	text-align: left;
	width: 110px;
	color: #666666;
	font-size: 12px;
}
.odd > td {
	text-align: left;
	width: 110px;
	color: #666666;
	font-size: 12px;
}
.last.odd > td {
	text-align: left;
	width: 110px;
	color: #666666;
	font-size: 12px;
}
.etabs {
	margin: 0;
	padding: 0;
	border-bottom:0px solid #8B0000;
}
.etabs2 {
	margin: 0;
	padding: 0;
	border-bottom:0px solid ##03C;
}
.tab-container .panel-container {
	background: #fff;/*border: solid #F6E5CB 15px;border: solid #F67F2B 15px;*/
	padding: 10px;
	border-radius:0 4px 4px 4px;
	-moz-border-radius: 0px 4px 4px 4px;
	-webkit-border-radius: 0px 4px 4px 4px;
}
.contenitore-dati-scheda-prodotto .etabs {
	border-bottom:10px solid #8B0000;
}
.tab {
	display: inline-block;
	zoom:1;
	background: #eaeaea; /*border: solid 1px #999;*/
	border-bottom: none;
	border-radius:4px 4px 0 0;
	-moz-border-radius: 4px 4px 0 0;
	-webkit-border-radius: 4px 4px 0 0;
	color: #3f3f3f;   /*left: -435px !important*/
	position: relative;
}
.tab2 {
	display: inline-block;
	zoom:1;
	background: #eaeaea; /*border: solid 1px #999;*/
	border-bottom: none;
	border-radius:4px 4px 0 0;
	-moz-border-radius: 4px 4px 0 0;
	-webkit-border-radius: 4px 4px 0 0;
	color: #3f3f3f;   /*left: -435px !important*/
	position: relative;
}
.tab a {
	font-size: 14px;
	line-height: 2.5em;
	display: block;
	padding: 0 25px;
	font-weight:bold;
	outline: none;
	color: #3f3f3f;
	text-decoration:none; /*text-shadow:1px 1px 1px #333333*/
}
.tab2 a {
	font-size: 14px;
	line-height: 2.5em;
	display: block;
	padding: 0 5px;
	font-weight:bold;
	outline: none;
	color: #3f3f3f;
	text-decoration:none; /*text-shadow:1px 1px 1px #333333*/
}
.tab a:hover {
	text-decoration: underline;
	background:#F00;
}
.tab.active {/*background: #8B0000;*/
	background-image:url(https://www.vosbesoinsinformatiques.com/wp-content/themes/blogolifepro/images/red/line.png);
	color: #fff;
	position: relative;
	border-color: #57B5CF; /*left: -435px;*/
}
.active2 {
	background: #CCC;
	color: #fff;
	position: relative;
	border-color: #57B5CF; /*left: -435px;*/
}
.tab a.active {
	font-weight: bold;
	color:#fff
}
#product-reviews h2 {
	font-family: Arial, Helvetica, sans-serif;
	font-weight:normal;
	color:#0083c5;
	font-size:24px
}
.panel-container h2 {
	color:#0083C5;
	font-family:'Roboto', Arial, Helvetica, sans-serif;
	font-weight:normal;
	font-size:18px;
	display:block;
	padding:4px 0 8px 0
}
.panel-container p {
	font-size:14px;
	color:#666;
	line-height:1.3em
}
.panel-container h3 {
	color:#8B0000;
	font-family:'lola';
	font-weight:normal;
	font-size:15px;
	display:block;
	padding:0 0 2px 0
}
.cur_on {
	border:1px solid #8B0000!important
}
.consigli h2 {
	display:block;
	color:#8B0000;
	font-family:'lola';
	font-weight:normal;
	padding:6px 10px 1px;
	font-size:23px;
}
.product-attribute-specs-table tr th {
	border-bottom: 1px solid #ccc;
	color: #333;
}
.product-attribute-specs-table tr td {
	border-bottom: 1px solid #ccc;
}
.image-neo {
	width:800px;
	height:608px;
}
.image-scout {
	width:800px;
	height:502px;
}
.description2 {
	margin-left:50px;
}/* BEGIN Ads div*/
.div_description {
	width:220px;
	height:270px;
	border:#CCC solid 1px;
}
.description_article {
	width:220px;
	height:50px;
	background-color:#F4F4F7;
	margin-top:-50px;
	font-family: 'Courier New', Courier, monospace;
	font-weight:normal;
	color:#333;
	font-size:14px;
	font-weight:bold;
	border:#CCC solid 1px;
	font-weight: bold;
}
.img-ads {
	width:220px;
	height:220px;
}
.voir {
	border:#CCC solid 1px;
	width:60px;
	text-align:center;
	color: #fff;
	border-color: #666; /*left: -435px;*/
	font-family: 'Courier New', Courier, monospace;
	font-weight:normal;
	color:#333;
font-size:14px-moz-border-radius: 10px;
	-webkit-border-radius: 10px;
	border-radius: 10px;
	background-color:#999;
	color:#FFF;
}
.voir a {
	text-decoration:none;
	color:#FFF;
	font-weight:bold;
}
.voir:hover {
	text-decoration:none;
	color:#FFF;
	font-weight:bold;
	background-color:#333;
}
.title_description {
	font-size:12px;
	font-weight:bold;
}
.price {
	color:#009;
	font-weight:bold;
	font-size:14px;
	font-family:'Comic Sans MS', cursive;
}/* End Ads div*/
.tab21 {
	display: inline-block;
	zoom:1;
	background: #eaeaea; /*border: solid 1px #999;*/
	border-bottom: none;
	border-radius:4px 4px 0 0;
	-moz-border-radius: 4px 4px 0 0;
	-webkit-border-radius: 4px 4px 0 0;
	color: #3f3f3f;   /*left: -435px !important*/
	position: relative;
}
.titre_h {
	text-shadow:0 0 2px #000,0 0 30px #000,0px 0px 5px #000, 0 0 150px #000;color:#fff;
	font-weight:bold;
	background: #fff;/*border: solid #F6E5CB 1px;*/
	padding: 10px;
	border-radius:0 4px 4px 4px;
	-moz-border-radius: 0px 4px 4px 4px;
	-webkit-border-radius: 0px 4px 4px 4px;/*border: solid #F67F2B 15px;*/
	background-color:#fff;
	text-align:center;
	
}
</style>
<!-- Facebook Pixel Code -->
<script>!function(f,b,e,v,n,t,s){if(f.fbq)return;n=f.fbq=function(){n.callMethod?n.callMethod.apply(n,arguments):n.queue.push(arguments)};if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';n.queue=[];t=b.createElement(e);t.async=!0;t.src=v;s=b.getElementsByTagName(e)[0];s.parentNode.insertBefore(t,s)}(window, document,'script','https://connect.facebook.net/en_US/fbevents.js');fbq('init', '221739482831230');fbq('track', 'PageView');</script>
<noscript>
<img height='1' width='1' style='display:none'src='https://www.facebook.com/tr?id=221739482831230&ev=PageView&noscript=1'/>
</noscript>
<!-- End Facebook Pixel Code -->
</head>
<body>
<div class='titre_h'><h1>Implémentez un modèle de scoring</h1></div>
<div id='tab-container' class='tab-container'>
  <ul class='etabs'>
    <li class='tab active'><a href='#tab-description' class='active'>Introduction</a></li>
    <!--    <li class='tab' id='upsell-tab'><a href='#tab-similar'>Prodotti Simili</a></li><li class='tab hidden-desktop' id='social-tab'><a href='#tab-social'>Condividere</a></li>-->
  </ul>
  <div class='bandeau-description'>
    <div class='description'>
      <div class='panel-container'>
        <div id='tab-description' style='display: block;' class='active'>
          <ul class='etabs2'>
            <li class='tab2 active'><strong><span class='tupper'><a href='#' class='active2'>Contexte :</a></span></strong> </li>
          </ul>
          <p>L’entreprise «HOME CREDIT» souhaite mettre en oeuvre un outil de «scoring crédit» pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).<br>
            Prêt à dépenser décide de développer un Dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.</p>
          <ul class='etabs2'>
            <li class='tab2 active'><strong><span class='tupper'><a href='#' class='active2'>Mossion :</a></span></strong></li>
          </ul>
          <div class='description2'>
            <p align='left'><img src='https://www.icone-png.com/png/29/29379.png' width='20px' height='20px' align='top' /> Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.</p>
            <p align='left'><img src='https://www.icone-png.com/png/29/29379.png' width='20px' height='20px' align='top' /> Construire un Dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
  <p>&nbsp;</p>
  <ul class='etabs'>
    <li class='tab active'><a href='#etabs' class='active'>Évaluation et découverte des données</a></li>
    <!--    <li class='tab' id='upsell-tab'><a href='#tab-similar'>Prodotti Simili</a></li><li class='tab hidden-desktop' id='social-tab'><a href='#tab-social'>Condividere</a></li>-->
  </ul>
  <div class='bandeau-description' width='100%'>&nbsp;
    <table width='100%' class='product-attribute-specs-table' id='product-attribute-specs-table'>
      <tbody>
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
      </tbody>
    </table>
  </div>
</div>
</body>
</html>

